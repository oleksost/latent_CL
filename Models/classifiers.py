from collections import namedtuple
from dataclasses import dataclass,field
from typing import Callable, Iterable, List, Optional, OrderedDict, Tuple, Union
# from clip.model import CLIP
from sklearn import neighbors
from torch import nn
import numpy as np
import torch
import os
import copy
import clip
import timm       
# import vissl
from PIL import Image  
from functools import partial  
import hashlib
from fvcore.nn import parameter_count_table
from torchvision.models.resnet import resnet18
from simple_parsing import choice
from torchvision.transforms.transforms import Lambda
from torchvision import models as torch_models
from torchvision import transforms   
from dataclasses_json import dataclass_json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification
from .helper import torch_NN, bn_eval, getKeysByValue
# from .StreamingLDA import StreamingLDA
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.nn.parallel import DistributedDataParallel as DDP

sklearn_classifiers = ['logistic_regression','random_forrest', 'knn']

#defines default behaviour for generate_cv_args function that generates cross validation arguments, useful for doing per task cross validation

@dataclass_json
@dataclass
class ClassifierOptions:
    def generate_cv_args(self):
        args = copy.copy(self)
        yield args
    @property
    def md5(self):
        self_copy = copy.copy(self)
        return hashlib.md5(str(self_copy).encode('utf-8')).hexdigest()


# Nearest Prototype (Similar to ICARL)
class NMC_Classifier(nn.Module):      
    """ Custom Linear layer but mimics a standard linear layer """
    
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        pass

    def __init__(self, size_in, size_out, device='cpu', *args, **kwargs):
        super().__init__()
        self.device=device               
        self.size_in, self.size_out = size_in, size_out
        self.data = torch.zeros(0, size_in)
        self.labels = torch.zeros(0)
        self.register_buffer('weight', torch.zeros(size_out, size_in))  # mean layer
        self.register_buffer('nb_inst', torch.zeros(size_out))
        self.register_buffer('_initiated', torch.Tensor([0.]))

    @property
    def initiated(self):
        if self._initiated.item()==0.:
            return False
        else:
            return True
    @initiated.setter
    def initiated(self, value:bool):
        if value:
            self._initiated=torch.Tensor([1.])
        else:
            self._initiated=torch.Tensor([0.])


    def __call__(self, x, y=None, epoch=None, *args, **kwargs):
        self.to('cpu')
        x=x.to('cpu')
        o = self.forward(x)
        if self.training and y is not None and epoch is not None:
            assert y is not None
            assert epoch is not None
            self.accumulate(x, y, epoch)
        return o

    def forward(self, x):
        data = x.detach().cpu()  # no backprop possible

        assert not torch.isnan(data).any()

        if self.initiated:
            # torch.cdist(c * b, d * b) -> c*d
            out = torch.cdist(data, self.weight)
            # convert smaller is better into bigger in better
            out = out * -1
        else:
            # if mean are not initiate we return random predition
            out = torch.randn((data.shape[0], self.size_out)).to(self.device)
        return out.to(self.device)

    def update(self, epoch=0):
        pass

    @torch.no_grad()
    def accumulate(self, x, y, epoch=0):
        if epoch == 0:
            self.data = x.view(-1, self.size_in).cpu()
            self.labels = y
            for i in range(self.size_out):
                indexes = torch.where(self.labels == i)[0]
                self.weight[i] = (self.weight[i] * (1.0 * self.nb_inst[i]) + self.data[indexes].sum(0))
                self.nb_inst[i] += len(indexes)
                if self.nb_inst[i] != 0:
                    self.weight[i] = self.weight[i] / (1.0 * self.nb_inst[i])

            self.data = torch.zeros(0, self.size_in)
            self.labels = torch.zeros(0)
            self.initiated = True

        assert not torch.isnan(self.weight).any()

    def expand(self, size_out):
        self.size_out=size_out
        weight = torch.zeros(self.size_out, self.size_in)
        weight[:self.weight.shape[0]]=self.weight
        self.register_buffer('weight', weight)  # mean layer
        nb_inst = torch.zeros(size_out)
        nb_inst[:self.nb_inst.shape[0]]=self.nb_inst
        self.register_buffer('nb_inst', nb_inst)
        # pass

class BiT_classifier(nn.Module):
  def __init__(self, in_size:int, n_classes:int, n_groups:int=32):
      super().__init__()
      self.model = nn.Sequential(OrderedDict([
          ('gn', nn.GroupNorm(n_groups, in_size)),
          ('relu1', nn.ReLU(inplace=True)),
          ('conv2', nn.AdaptiveAvgPool2d(output_size=1)),   
          ('relu2', nn.Conv2d(in_size, n_classes, kernel_size=(1,1), stride=(1,1)))
        ]))
  def forward(self, x, *args,**kwargs):
    if len(x.shape)==2:   
      #this classifier expects 4d tensor
      x=x.unsqueeze(2).unsqueeze(3)
    return self.model(x)

class CLIPZeroShotClassifier(nn.Module):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        pass
    def __init__(self, text_encodings:torch.Tensor) -> None:
        super().__init__()
        # self.encoder = model
        # self.classes = clip.tokenize(class_descriptions)
        # with torch.no_grad():
        #     self.text_features = self.encoder.encode_text(self.classes)
        self.text_features = text_encodings.float().detach()
    def __call__(self, image_features:torch.Tensor, *args,**kwargs):
        with torch.no_grad():
            # logits_per_image, logits_per_text = self.encoder(image_features, self.text_features)
            # Pick the top 5 most similar labels for the image
            image_features=image_features.float()        
            image_features /= image_features.norm(dim=-1, keepdim=True) 
            text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            # values, indices = similarity[0].topk(5)
            # probs = similarity.softmax(dim=-1).cpu().numpy()
        return similarity

class Classifier_nn(nn.Module):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        cls_n_hidden: int = 0 # -
        cls_hidden_size: Optional[int] = None #-    
        cls_use_bn: bool = False # -

    def __init__(self, in_dim, n_classes, options:Options=None, *args, **kwargs) -> None:
        super().__init__()
        if options is None:
            self.options=Classifier_nn.Options()
        else:
            self.options = options    
        if self.options.cls_hidden_size is None:
            hidden_size= in_dim*2
        else:
            hidden_size= self.options.cls_hidden_size
        if self.options.cls_n_hidden>0:
            hidden = [hidden_size for n in range(self.options.cls_n_hidden)]
        else:
            hidden=[]
        self.classifier = nn.Sequential(nn.Flatten(), torch_NN(in_dim,n_classes,hidden, use_bn=self.options.cls_use_bn))
    def forward(self, x, *args,**kwargs):
        return self.classifier(x)

class Classifier_BiT(nn.Module):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        n_groups: int = 32 #-
        # def generate_cv_args(self):
        #     n_groups = np.arange(12,64, 6)
        #     for n_group in n_groups:
        #         args=copy.copy(self)
        #         args.n_groups=n_group
        #         yield args

    def __init__(self,in_dim,n_classes, options:Options=None):
        super().__init__()
        if options is None:
            self.options = Classifier_BiT.Options()
        else:
            self.options = options
        self.classifier=BiT_classifier(in_dim, n_classes, self.options.n_groups)
    def forward(self,x, *args,**kwargs):
        return self.classifier(x)

class RandomForest_Classifier(RandomForestClassifier):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        pass
    def __init__(self, *args, options:Options=None, **kwargs):
        if options is None:
            options = RandomForest_Classifier.Options()
        else:
            self.options = options
        super().__init__(*args, **kwargs)

class LogisticRegression_Classifier(LogisticRegression):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        C:float=0.316
        def generate_cv_args(self):
            Cs = np.arange(0.1, 0.5, 0.05)
            for C in Cs:
                args=copy.copy(self)
                args.C=C
                yield args
    def __init__(self, *args, options:Options=None, **kwargs):
        if options is None:
            self.options = LogisticRegression_Classifier.Options()
        else:
            self.options=options
        super().__init__(*args, C=self.options.C, **kwargs)
        
class KNeighbors_Classifier(KNeighborsClassifier):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        n_neighbors:int=5 #-                    
        neighbours:List =field(default_factory=lambda:[5,10,20,50,100])
        def generate_cv_args(self):
            n_neighbors = [5,10,20,50,100]
            for n in self.neighbours:
                args=copy.copy(self)
                args.n_neighbors=n
                yield args

    def __init__(self, *args, options:Options=None, **kwargs):
        if options is None:
            self.options = KNeighbors_Classifier.Options()
        else:
            self.options=options
        super().__init__(*args, n_neighbors=self.options.n_neighbors, **kwargs)

class SLDA_Classifier(nn.Module):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        pass

    def __init__(self, size_in, size_out, test_batch_size=1024, shrinkage_param=1e-4,
                 streaming_update_sigma=True, device='cpu', *args, **kwargs):
        # if options is None:
        #     self.options = NMC_Classifier.Options()
        # else:
        #     self.options=options
        super().__init__()  # *args, **kwargs)
        self.size_in = size_in
        self.size_out = size_out

        # SLDA parameters
        self.device = device
        # self.device = 'cuda'
        self.input_shape = self.size_in
        self.num_classes = self.size_out
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        # setup weights for SLDA
        self.register_buffer('muK', torch.zeros((self.num_classes, self.input_shape)).to(self.device))
        self.register_buffer('cK', torch.zeros(self.num_classes).to(self.device))
        self.register_buffer('Sigma', torch.ones((self.input_shape, self.input_shape)).to(self.device))
        self.register_buffer('num_updates', torch.zeros(1).to(self.device))
        self.register_buffer('_initiated', torch.Tensor([0.]).to(self.device))
        self.register_buffer('Lambda', torch.zeros_like(self.Sigma).to(self.device))
        # self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    @property
    def initiated(self):
        if self._initiated.item()==0.:
            return False
        else:
            return True
    @initiated.setter
    def initiated(self, value:bool):
        if value:
            self._initiated=torch.Tensor([1.])
        else:
            self._initiated=torch.Tensor([0.])

    def __call__(self, x, y=None, epoch=None, *args, **kwargs):
        o = self.forward(x)
        if self.training and y is not None and epoch is not None:
            assert y is not None
            assert epoch is not None
            self.accumulate(x, y, epoch)
        return o

    def forward(self, x):
        if self.initiated:
            x = self.predict(x)
        else:
            x = torch.randn((x.shape[0], self.size_out)).to(self.device)

        return x.to(self.device)

    def accumulate(self, x, y, epoch=0):
        if epoch == 0:
            self.initiated = True
            x = x.view(-1, self.size_in)
            for i in range(len(y)):
                self.fit(x[i], y[i])

    def expand(self, size_out):
        self.size_out=size_out   
        self.num_classes=self.size_out

        muK = torch.zeros((self.num_classes, self.input_shape))
        cK = torch.zeros(self.num_classes)
        
        muK[:self.muK.shape[0]]=self.muK
        cK[:self.cK.shape[0]]=self.cK
        self.register_buffer('muK', muK.to(self.device))
        self.register_buffer('cK', cK.to(self.device))

    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        with torch.no_grad():
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                #print('\nFirst predict since model update...computing Lambda matrix...')
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                        self.device))
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                scores[start:end, :] = torch.matmul(x, W) - c

            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze().long()

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        print('\nEstimating initial covariance matrix...')
        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)

    @torch.no_grad()
    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """

        x = x.view(-1).to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)


        # covariance updates
        if self.streaming_update_sigma:
            x_minus_mu = (x - self.muK[y])
            mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
            delta = mult * self.num_updates / (self.num_updates + 1)
            self.Sigma = (self.num_updates * self.Sigma + delta)
            self.Sigma = self.Sigma / (self.num_updates[0] + 1)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates[0] += 1

class WeightNorm_Classifier(nn.Module):
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        bias: bool = 1

    def __init__(self, in_dim, n_classes, *args, options:Options=None, **kwargs):
        super().__init__()
        if options is None:
            self.options = WeightNorm_Classifier.Options()
        else:
            self.options=options
        self.size_in, self.size_out = in_dim, n_classes
        self.weight = nn.Parameter(torch.Tensor(n_classes, in_dim))
        if self.options.bias:
            self.bias = nn.Parameter(torch.zeros(n_classes))
        else:
            self.bias = None

        # initialize weights
        nn.init.kaiming_normal_(self.weight)  # weight init
        # super().__init__(*args, **kwargs)
    def forward(self, x, *args, **kwargs):
        return torch.nn.functional.linear(x, self.weight / torch.norm(self.weight, dim=1, keepdim=True), self.bias)
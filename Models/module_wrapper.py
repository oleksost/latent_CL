from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, OrderedDict, Tuple, Union
from torch import nn
import torch
import copy
from fvcore.nn import parameter_count_table
from torchvision.models.resnet import resnet18
from simple_parsing import choice
from torchvision import transforms   
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from Models.classifiers import NMC_Classifier, SLDA_Classifier
from .helper import bn_eval

class TorchModuleWrapper(nn.Module):   
    """
    Wraps the nn.Module to define additional functionality.
    """
    def __init__(self, feature_extractor:nn.Module, classifiers:nn.Module=None,
                function: Callable=None, keep_bn_in_eval_after_freeze:bool=False,
                flatten_image:bool=False,  flatten_features=False, optimizer_name: str = None, 
                optimizer_name_classifier: str=None, lr: float = None, lr_classifier: float = None, momentum:float=None, weight_decay:float=None):

        super().__init__()
        self.register_buffer('_mapping',torch.zeros(1)) #for remaping class labels if order is permuted of for multihead scenario
        self.flatten_image=flatten_image
        self.flatten_features=flatten_features
        self.function=function
        self.feature_extractor=feature_extractor
        self.classifiers = classifiers
        self.keep_bn_in_eval_after_freeze = keep_bn_in_eval_after_freeze
        self.optimizer: torch.optim.Optimizer = None
        self.bn_eval_layer=None

        #optimizer params
        self.optimizer_name: str = optimizer_name    
        self.optimizer_name_classifier: str = optimizer_name_classifier


        self.lr = lr                
        self.weight_decay = weight_decay
        self.lr_classifier = lr_classifier
        self.momentum = momentum

        self.optimizer = None
        self.optimizer_classifier = None
    
    # def update_mapping(self, task_id, new_classes):   
    #     _, mapping = remap_class_vector(new_classes, mapping)
    #     pass

    @property
    def mapping(self):
        if torch.sum(self._mapping)==0:
            return None
        return self._mapping

    @mapping.setter
    def mapping(self,v):
        if isinstance(v, torch.Tensor):
            self._mapping=v
        else:
            v=torch.tensor(v, dtype=torch.int)
            self._mapping=v      
    
    def add_classifier_head(self, classifier):  
        assert isinstance(self.classifiers, Iterable) 
        self.classifiers.append(classifier)
    
    def expand_classifier(self, new_classifier:nn.Module):
        assert isinstance(self.classifiers, nn.Module)
        #copy weights 
        if isinstance(self.classifiers, NMC_Classifier):
            assert isinstance(new_classifier, NMC_Classifier)
            self.classifiers.expand(new_classifier.size_out)
        elif isinstance(self.classifiers, SLDA_Classifier):
            assert isinstance(new_classifier, SLDA_Classifier)
            self.classifiers.expand(new_classifier.size_out)
            
        else:
            for m_from, m_to in zip(self.classifiers.modules(), new_classifier.modules()):
                is_linear = isinstance(m_to, torch.nn.Linear)
                is_conv = isinstance(m_to, torch.nn.Conv2d)
                is_bn = isinstance(m_to, torch.nn.BatchNorm2d) or isinstance(m_to, torch.nn.GroupNorm)
                if is_linear or is_bn:
                    if m_to.weight.data.shape==m_from.weight.data.shape:
                        m_to.weight.data = m_from.weight.data.clone()
                    else:
                        m_to.weight.data[:m_from.weight.data.shape[0]] = m_from.weight.data.clone()
                    if m_to.bias.data.shape==m_from.bias.data.shape:
                        m_to.bias.data = m_from.bias.data.clone()
                    else:
                        m_to.bias.data[:m_from.bias.data.shape[0]] = m_from.bias.data.clone()
                elif is_conv:
                    if m_to.weight.data.shape==m_from.weight.data.shape:
                        m_to.weight.data = m_from.weight.data.clone()
                    else:
                        m_to.weight.data[:m_from.weight.data.shape[0]] = m_from.weight.data.clone()
                    if m_to.bias.data.shape==m_from.bias.data.shape:
                        m_to.bias.data = m_from.bias.data.clone()
                    else:
                        m_to.bias.data[:m_from.bias.data.shape[0]] = m_from.bias.data.clone()
            self.classifiers=new_classifier
    
    def train(self, *args, **kwargs):   
        if self.keep_bn_in_eval_after_freeze:
          r = super().train(*args, **kwargs)
          if self.bn_eval_layer is None:
            bn_eval(self.feature_extractor)
          else:
            bn_eval(self.feature_extractor.feature_extractor.vit.encoder.layer[:self.bn_eval_layer])
          return r
        else:
          return super().train(*args, **kwargs)

    def forward(self, x:torch.Tensor, task_id=None, *args, **kwargs):
        out = x
        if self.flatten_image:
            out = out.flatten(1)
        if self.feature_extractor is not None:
            if self.function is None:  
                out = self.feature_extractor(out)
            else:
                try:
                    out = self.function(self.feature_extractor, out)
                except:
                    out = self.function(out)
        #features
        if self.flatten_features:         
            out = torch.flatten(out,start_dim=1).to(torch.float32) 
        if self.classifiers is not None:
            if task_id is None:
                if isinstance(self.classifiers, nn.ModuleList):
                    out = self.classifiers[-1](out, *args,**kwargs)
                else:
                    out = self.classifiers(out, *args, **kwargs)
            else:
                if isinstance(self.classifiers, nn.ModuleList):
                    out = self.classifiers[task_id](out,*args, **kwargs)
                else:
                    out = self.classifiers(out,*args,**kwargs)
            return out.squeeze()
        return out
    
    def set_optimizer(self): 
        if self.optimizer_name_classifier is None: 
            params = filter(lambda x: x.requires_grad, self.parameters())
            if not len(list(params))==0:
                assert self.lr is not None
                assert self.optimizer_name is not None
                if self.optimizer_name=='adam':
                    self.optimizer= torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
                elif self.optimizer_name=='sgd':
                    self.optimizer= torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            else:
                self.optimizer=None
        else:
            #seperate optimizers for classifier and feature extractor
            params_fe = filter(lambda x: x.requires_grad, self.feature_extractor())
            if not len(list(params_fe))==0:
                assert self.lr is not None
                assert self.optimizer_name is not None
                if self.optimizer_name=='adam':
                    self.optimizer= torch.optim.Adam(filter(lambda x: x.requires_grad, self.feature_extractor()), lr=self.lr)
                elif self.optimizer_name=='sgd':
                    self.optimizer= torch.optim.SGD(filter(lambda x: x.requires_grad, self.feature_extractor()), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            else:
                self.optimizer=None

            params_cls = filter(lambda x: x.requires_grad, self.feature_extractor())
            if not len(list(params_cls))==0:
                assert self.lr_classifier is not None
                assert self.optimizer_name_classifier is not None
                self.optimizer_classifier= torch.optim.Adam(filter(lambda x: x.requires_grad, self.feature_extractor()), lr=self.lr)
            else:
                self.optimizer_classifier=None
        
    def step(self):    
        if self.optimizer is not None:
            self.optimizer.step()
        if self.optimizer_classifier is not None:
            self.optimizer_classifier.step()
        
    def create_checkpoint(self):
        state_dict=copy.deepcopy(self.state_dict())
        classifiers = None
        if  any([isinstance(self.classifiers,cl) for cl in [LogisticRegression, RandomForestClassifier, KNeighborsClassifier]]):
            classifiers = copy.deepcopy(self.classifiers)
        return (state_dict,classifiers)
    
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        #addaprt the shape of _maapings
        if '_mapping' in state_dict.keys():
            self._mapping = torch.zeros_like(state_dict['_mapping'])
        return super().load_state_dict(state_dict, strict=strict)
    
    def load_checkpoint(self, checkpoint:Tuple):
        state_dict, classifiers = checkpoint
        self.load_state_dict(state_dict)
        if classifiers is not None:
            self.classifiers = classifiers
        
    def freeze_feature_extractor(self, freeze=True):
        if self.feature_extractor is not None:
            for p in self.feature_extractor.parameters():
                p.requires_grad=not freeze
                p.grad=None
            if freeze and self.keep_bn_in_eval_after_freeze:
                    bn_eval(self.feature_extractor)
                # self.keep_bn_in_eval_after_freeze=True
            # else:
            #     self.keep_bn_in_eval_after_freeze=False
        self.set_optimizer()

    def unfreeze_bn(self):        
        def unfreeze_norms(module:torch.nn.Module):
            for layer in module.children():      
                if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.BatchNorm3d) or isinstance(layer, torch.nn.LayerNorm):
                    for p in layer.parameters():
                        p.requires_grad=True
                elif isinstance(layer.children(), Iterable):
                    unfreeze_norms(layer)
        unfreeze_norms(self)
        self.set_optimizer()

    def unfreeze_first(self):  
        layer=self.feature_extractor.feature_extractor.vit.embeddings
        for p in layer.parameters():
                p.requires_grad=True
        self.set_optimizer()
    
    def freeze_vit_untill_layer(self, layer=0):
        #freeze input layer and positional embedings
        for p in self.feature_extractor.feature_extractor.vit.embeddings.parameters():
            p.requires_grad=False
            p.grad=None  
        for i in range(layer):   
             for p in self.feature_extractor.feature_extractor.vit.encoder.layer[:i].parameters():
                p.requires_grad=False
                p.grad=None  
        self.keep_bn_in_eval_after_freeze=1
        self.bn_eval_layer=layer
        bn_eval(self.feature_extractor.feature_extractor.vit.encoder.layer[:layer])
        self.set_optimizer()

from typing import Callable, Iterable
import torch
import copy
from dataclasses import dataclass
import numpy as np 
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from continuum.tasks import TaskType
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_balanced_sampler(taskset, log=False):
    """Create a sampler that will balance the dataset.
    You should give the returned sampler to the dataloader with the argument `sampler`.
    :param taskset: A pytorch dataset that implement the TaskSet interface.
    :param log: Use a log weights. If enabled, there will still be imbalance but
                on the other hand, the oversampling/downsampling won't be as violent.
    :return: A PyTorch sampler.
    """
    if taskset.data_type in (TaskType.SEGMENTATION, TaskType.OBJ_DETECTION, TaskType.TEXT):
        raise NotImplementedError(
            "Samplers are not yet available for the "
            f"{taskset.data_type} type."
        )

    y = taskset.get_raw_samples()[1]
    nb_per_class = np.bincount(y)
    weights_per_class = 1 / nb_per_class
    if log:
        weights_per_class = np.log(weights_per_class)
        weights_per_class = 1 - (weights_per_class / np.sum(weights_per_class))

    weights = weights_per_class[y]

    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(taskset))

'''
Get a list of keys from dictionary which has the given value
'''
def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def bn_eval(module:torch.nn.Module, freeze=True):
    for layer in module.children():      
        if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.BatchNorm3d):
          if freeze:
            layer.eval()
          else:
            layer.train()
        elif isinstance(layer.children(), Iterable):
            bn_eval(layer, freeze)

#copied from https://github.com/FerranAlet/modular-metalearning
class torch_NN(nn.Module):
  '''
  Mimic the pytorch-maml/src/ominglot_net.py structure
  '''
  def __init__(self, inp=1, out=1, hidden=[], final_act='affine', bias=True, loss_fn=None, use_bn=False):
    super(torch_NN, self).__init__()
    self.inp = inp
    self.use_bn=use_bn
    self.dummy_inp = torch.randn(8, inp, device=device)
    self.out = out
    self.num_layers = len(hidden) + 1
    self.final_act = final_act
    key_words = []
    for i in range(self.num_layers):
      key_words.append('fc_'+str(i))
      if i < self.num_layers-1: #final_act may not be a relu
        key_words.append('relu_'+str(i))
        if self.use_bn:
          key_words.append('bn_'+str(i))

    def module_from_name(name):  
      # if self.num_layers >10: raise NotImplementedError
      #TODO: allow more than 10 layers, put '_' and use split
      num = int(name.split('_')[1])
      typ = name.split('_')[0]
      if typ == 'fc':
        inp = self.inp if num==0 else hidden[num-1]
        out = self.out if num+1==self.num_layers else hidden[num]
        return nn.Linear(inp, out, bias=bias)
      elif typ=='relu':
        return nn.ReLU() #removed inplace
      elif typ=='bn':
        return nn.BatchNorm1d(hidden[num-1])  
      else: raise NotImplementedError

    self.add_module('features', nn.Sequential(OrderedDict([
      (name, module_from_name(name)) for name in key_words])))

    if self.final_act == 'sigmoid': self.add_module('fa', nn.Sigmoid())
    elif self.final_act == 'exp': self.add_module('fa', exponential())
    elif self.final_act == 'affine': self.add_module('fa', nn.Sequential())
    elif self.final_act == 'relu': self.add_module('fa', nn.ReLU())
    elif self.final_act == 'tanh': self.add_module('fa', nn.Tanh())
    else: raise NotImplementedError

  def dummy_forward_pass(self):
    '''
    Dummy forward pass to be able to backpropagate to activate gradient hooks
    '''
    return torch.mean(self.forward(self.dummy_inp))

  def forward(self, x, weights=None, prefix='', **kwargs):
    '''
    Runs the net forward; if weights are None it uses 'self' layers,
    otherwise keeps the structure and uses 'weights' instead.
    '''
    if weights is None:
      x = self.features(x)
      x = self.fa(x)
    else:
      for i in range(self.num_layers):
        x = linear(x, weights[prefix+'fc'+str(i)+'.weight'],
                weights[prefix+'fc'+str(i)+'.bias'])
        if i < self.num_layers-1: x = relu(x)
      x = self.fa(x)
    return x

  def net_forward(self, x, weights=None):
    return self.forward(x, weights)

  #pytorch-maml's init_weights not implemented; no need right now.

  def copy_weights(self, net):
    '''Set this module's weights to be the same as those of 'net' '''
    for m_from, m_to in zip(net.modules(), self.modules()):
      if (isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d)
          or isinstance(m_to, nn.BatchNorm2d)):
        m_to.weight.data = m_from.weight.data.clone()
        if m_to.bias is not None:
            m_to.bias.data = m_from.bias.data.clone()


'''
Functional definitions of common layers
Useful for when weights are exposed rather
than being contained in modules
'''

def linear(input, weight, bias=None):
    if bias is None:
        return F.linear(input, weight.cuda())
    else:
        return F.linear(input, weight.cuda(), bias.cuda())

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight.cuda(), bias.cuda(), stride, padding, dilation, groups)

def relu(input):
    return F.threshold(input, 0, 0, inplace=True)

def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)

def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    # This hack only works when momentum is 1 and avoids needing to track running stats
    # by substuting dummy variables
    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).cuda()
    running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).cuda()
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def bilinear_upsample(in_, factor):
    return F.upsample(in_, None, factor, 'bilinear')

def log_softmax(input):
    return F.log_softmax(input)

class exponential(nn.Module):
    def __init__(self):
        super(exponential, self).__init__()
    def forward(self, x):
        return torch.exp(x)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden_two = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden_3 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden_two(x))  # activation function for hidden layer
        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
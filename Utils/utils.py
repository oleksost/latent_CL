import random
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar, Union, Optional
import torch
import numpy as np
import wandb
import scipy
import copy
from tqdm import tqdm
from tinydb import TinyDB, Query
from collections import OrderedDict       
from torchnet.meter import AverageValueMeter
T = TypeVar("T")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from args import ArgsGenerator
from continuum.tasks import TaskType
from torch import nn
from continuum.tasks.base import BaseTaskSet
from torch.functional import Tensor
from continuum.tasks.task_set import TaskSet
from continuum.datasets import InMemoryDataset
from torch.utils.data import DataLoader, Dataset
from continuum.scenarios import OnlineFellowship
from continuum.scenarios import ContinualScenario
from continuum.rehearsal.memory import RehearsalMemory

import logging
from Models.classifiers import ClassifierOptions
from continuum.tasks import split_train_val
from continuum.tasks.h5_task_set import H5TaskSet
from torchvision.transforms.transforms import Compose, Lambda, Resize, ToTensor
from Models.model import (ModelContainer, ModelContainer_ER, TorchModuleWrapper)

class TinyDB_hp_tracker():   
    def __init__(self, db_location, key) -> None:
        self.hp_db = TinyDB(db_location)
        self.key=key
        self.query = Query()

    def load_hps_from_db(self, args_global:ArgsGenerator,args_model:ModelContainer.Options,args_classifier:ClassifierOptions):
        '''
        tries to load hyperparameters corresponding to this run from hp_database, 
        otherwise sets the flags to create new entry to hp_database later
        '''
        #use hyperparameter database
        result=self.hp_db.search(self.query.key == self.key)
        if len(result)==0:
            #no hp found
            assert args_global.task_level_cv
            assert args_global.keep_best_params_after_first_task
            return False,args_global,args_model,args_classifier
        else:
            #load hyperparams from db
            logging.info(f'Loading existing optimal hyperparams for this run (key {self.key}')
            invariant_global_args = args_global.get_hp_invariant_attributes()
            args_global = args_global.__class__.from_dict(result[0]['args']['args_global'])
            args_model = args_model.__class__.from_dict(result[0]['args']['args_model'])
            args_classifier = args_classifier.__class__.from_dict(result[0]['args']['args_classifier'])
            for key,attr in invariant_global_args.items():
                setattr(args_global,key,attr)
            args_global.task_level_cv=0

        return True,args_global,args_model,args_classifier

    def log_into_hp_ds(self, args_global:ArgsGenerator,best_args_model:ModelContainer.Options, best_args_classifier:ClassifierOptions):
        #save best parameters to hp database
        assert self.key is not None
        if len(self.hp_db.search(self.query.key == self.key))==0:
            #otherwise it was written meanwhile by other run
            assert best_args_classifier is not None
            assert best_args_model is not None
            self.hp_db.insert({'key':self.key, 'args':{'args_global':args_global.serializable_copy().to_dict(), 
                                'args_model':best_args_model.serializable_copy().to_dict(),
                                'args_classifier':best_args_classifier.to_dict()}})

 
def convert_to_task_set(args:ArgsGenerator, dataset_train:H5TaskSet)->TaskSet:
    x,y,t = dataset_train.get_raw_samples()   
    if len(y.shape)>1:
            y=y[0]
            t=t[0]   
    dataset_train = TaskSet(x,y,t, trsf=dataset_train.trsf, data_type=args.tasktype, target_trsf=dataset_train.target_trsf)
    return dataset_train


def prepare_dataloader(args:ArgsGenerator, dataset:Dataset, val_split=0.1, shuffle=False, num_workers=0, world_size=None, rank=None, transforms_val=None, sampler=None):
    if val_split>0:
        train_taskset, val_taskset = split_train_val(dataset, val_split=val_split)  
        if transforms_val is not None:
            #set validation transfroms 
            if isinstance(transforms_val, List): 
                transforms_val = Compose(transforms_val)
            val_taskset.trsf = transforms_val
        if world_size is not None and rank is not None:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)   
            train_loader = DataLoader(train_taskset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=num_workers, drop_last=False, sampler=sampler)
            val_loader = DataLoader(val_taskset, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=num_workers, drop_last=False, sampler=sampler)
        else:
            #TODO: this would not work in distributed setting 
            train_loader = DataLoader(train_taskset, batch_size=args.batch_size, shuffle=True, sampler=sampler)
            val_loader = DataLoader(val_taskset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)
        return train_loader,val_loader
    else:
        if world_size is not None and rank is not None:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, pin_memory=False, num_workers=num_workers, drop_last=False, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return loader, None

class FlopsMeter():
    def __init__(self, names: List) -> None:
        #meter for flops counting
        self.meters: Dict[str, SumMeter] = {}
        if len(names)>0:
            for name in names:
                self.meters[name]=SumMeter()
    
    def add(self, data: Union[Dict, List]={}):
        if isinstance(data, Dict):
            if len(data.keys())>0:  
                for k,v in data.items():
                    self.meters[k].add(v)
        else:
            for v,(k,meter) in zip(data, self.meters.items()):
                    meter.add(v)
    def log_flops(self, task=None):
        if task is None:
            for k,meter in self.meters.items():
                log_wandb({f'compute/{k}':meter.value()[0]})
        else:
            for k,meter in self.meters.items():
                log_wandb({f'compute/{k}_task{task}':meter.value()[0]})
    
    def value(self, function:Callable=None)->Dict[str,float]:
        if function is None:
            return {k:meter.value()[0] for k,meter in self.meters.items()}
        else:
            return {k:function(meter.value[0]) for k,meter in self.meters.items()}

  
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
    # max_nb_class = max(nb_per_class)
    # nb_classes = sum(nb_per_class!=0)

    weights_per_class = 1 / nb_per_class

    if log:
        weights_per_class = np.log(weights_per_class)      
        weights_per_class = 1 - (weights_per_class / np.sum(weights_per_class[(1-np.isinf(weights_per_class))==1]))

    weights = weights_per_class[y]
    prob_to_sample_class = [sum(weights[y==c])/sum(weights) for c in np.unique(y)]
    print('prob_to_sample_class',prob_to_sample_class/(sum(prob_to_sample_class)))
    entropy = scipy.stats.entropy(prob_to_sample_class, base=10)/np.log10(len(prob_to_sample_class))
    print('entropy class samnpling distribiution',entropy)
    log_wandb({'class_sampling_entropy':entropy})
    return torch.utils.data.sampler.WeightedRandomSampler(weights, int(len(taskset)), replacement=False)

def create_balanced_taskset(x,y,t,base_taskset, classes_current_task):
    class TaskSet_balanced(base_taskset.__class__):
        def __init__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, trsf: Union[Compose, List[Compose]], target_trsf: Optional[Union[Compose, List[Compose]]] = None, bounding_boxes: Optional[np.ndarray] = None, classes_current_task: np.ndarray=None, debug=False):
            super().__init__(x, y, t, trsf=trsf, target_trsf=target_trsf, bounding_boxes=bounding_boxes)
            self.new_task_indicies=np.where(np.in1d(y,classes_current_task))[0]
            self.new_task_indicies_copy=copy.copy(self.new_task_indicies)
            self.seen_idx=[]
            self.debug=debug
            
        def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
            self.seen_idx.append(index)
            # if self.debug:
            #     print('unseen from the task',len(self.new_task_indicies))     
            #     if len(self.new_task_indicies)==1:
            #         print(index)
            #     print('len seen',len(self.seen_idx)) 
            #     print('len unique seen',len(np.unique(self.seen_idx)))   
            self.new_task_indicies_copy=self.new_task_indicies_copy[np.where(1-(self.new_task_indicies_copy==index))[0]]
            if len(self.new_task_indicies_copy)>0:
                if len(self.seen_idx)==len(self):
                    self.new_task_indicies_copy=copy.copy(self.new_task_indicies)
                return super().__getitem__(index)
            else:
                #stop iteration if all new samples have been seen
                self.new_task_indicies_copy=copy.copy(self.new_task_indicies)
                raise StopIteration
    return TaskSet_balanced(x,y,t,trsf=base_taskset.trsf, target_trsf=base_taskset.target_trsf, classes_current_task=classes_current_task)
    

def oversample(mem_xm, mem_y, mem_t, n_per_class):   
    #oversample per class
    classes = np.unique(mem_y)
    counts = np.bincount(mem_y)
    n_oversample = n_per_class - counts[counts>0]
    def select_idxs(c, y, n):
        return np.random.choice(np.where(y==c)[0], n)
    for c, n in zip(classes, n_oversample):  
        selected_inds = select_idxs(c,mem_y,int(max(n,0)))  
        mem_xm = np.concatenate([mem_xm, mem_xm[selected_inds]])
        mem_y = np.concatenate([mem_y, mem_y[selected_inds]])
        mem_t = np.concatenate([mem_t, mem_t[selected_inds]])
    return mem_xm, mem_y, mem_t

def prepare_balanced_batch_train_loader(args:ArgsGenerator, buffer_train, current_task_set):
    if isinstance(buffer_train, RehearsalMemory):
        mem_x, mem_y, mem_t = buffer_train.get()
    elif isinstance(buffer_train, BaseTaskSet): 
        mem_x, mem_y, mem_t = buffer_train.get_raw_samples()
    if isinstance(x[0], str):
        args.tasktype=TaskType.IMAGE_PATH
    train_taskset = TaskSet(x=mem_x, y=mem_y, t=mem_t, target_trsf=current_task_set.train_taskset, trsf=train_taskset.trsf, data_type=args.tasktype)
    print(f'N samples per class {np.bincount(train_taskset.get_raw_samples()[1])}')
    return prepare_dataloader(args, train_taskset, val_split=0, shuffle=True)[0]

def prepare_balanced_train_loader(args:ArgsGenerator, buffer_train, current_task_set):
    if isinstance(buffer_train,BaseTaskSet):
        mem_xm, mem_y, mem_t = buffer_train.get_raw_samples()
    else:
        mem_xm, mem_y, mem_t = buffer_train.get() 
        #TODO: problem when comnining data from tasks with diffierent datatype into the replay buffer
        # if isinstance(mem_xm[0], str):
        #     tasktype=TaskType.IMAGE_PATH
        # else:
        #     if args.regime=='sample_ER':
        #         tasktype=TaskType.IMAGE_ARRAY
        #     else:
        #         tasktype=TaskType.TENSOR
        # if tasktype!=current_task_set.data_type:  
        #     replay_taskset = TaskSet(x=mem_xm, y=mem_y, t=mem_t, target_trsf=current_task_set.target_trsf, trsf=Compose([ToTensor(),Resize((100, 100))]), data_type=tasktype)
        #     mem_xm, mem_y, mem_t = replay_taskset.get_samples(np.arange(len(replay_taskset._x))) 

    new_classes = np.unique(current_task_set.get_raw_samples()[1])
    sampler = None
    shuffle = True

    #balance the current task
    x,y,t = current_task_set.get_raw_samples()
    x,y,t = oversample(x,y,t,max(np.bincount(y)))
    count_per_class = np.bincount(y)

    if max(count_per_class)>args.er_size_per_class:
        #oversample the replay buffer 
        # TaskSet
        current_task_set = TaskSet(x,y,t,trsf=current_task_set.trsf, target_trsf=current_task_set.target_trsf, data_type=current_task_set.data_type) #BaseTaskSet(x,y,t,trsf=current_task_set.trsf, target_trsf=current_task_set.target_trsf)
        avv_samples_per_class_new_task = count_per_class[count_per_class!=0].mean()
        mem_xm, mem_y, mem_t = oversample(mem_xm, mem_y, mem_t, avv_samples_per_class_new_task)
        current_task_set.add_samples(mem_xm, mem_y, mem_t)
        count_per_class = np.bincount(current_task_set._y)
        prob_to_sample_class = count_per_class[count_per_class!=0]/sum(count_per_class)

        print('prob_to_sample_class',prob_to_sample_class)
        entropy = scipy.stats.entropy(prob_to_sample_class, base=10)/np.log10(len(prob_to_sample_class))
        print('entropy class samnpling distribiution',entropy)
        log_wandb({'class_sampling_entropy':entropy})

    else:
        #undersample the replay buffer at each epoch   
        current_task_set = TaskSet(x,y,t,trsf=current_task_set.trsf, target_trsf=current_task_set.target_trsf, data_type=current_task_set.data_type) #
        current_task_set= create_balanced_taskset(x,y,t, current_task_set, classes_current_task=new_classes) #TaskSet_balanced(x,y,t,trsf=current_task_set.trsf, target_trsf=current_task_set.target_trsf, classes_current_task=new_classes, debug=args.debug)
        shuffle=False  
        current_task_set.add_samples(mem_xm, mem_y, mem_t)    
        sampler = get_balanced_sampler(current_task_set, log=False)    
    
    replay_dataloader, _ = prepare_dataloader(args, current_task_set, val_split=0, shuffle=shuffle, sampler=sampler)
    print(f'N samples per class {np.bincount(current_task_set.get_raw_samples()[1])}')
    return replay_dataloader

def prepare_randombuffer_train_loader(args:ArgsGenerator, train_taskset, buffer_train):
    ########## dev  
    x,y,t = train_taskset.get_raw_samples()     
    #at some point y hat a shape (1, dataset_size) (maybe different continuum branch)
    if len(y.shape)>1:
        y=y[0]
        t=t[0] 
    if isinstance(buffer_train, RehearsalMemory):
        mem_x, mem_y, mem_t = buffer_train.get()
    elif isinstance(buffer_train, BaseTaskSet): 
        mem_x, mem_y, mem_t = buffer_train.get_raw_samples()
    if isinstance(x[0], str):
        args.tasktype=TaskType.IMAGE_PATH
    train_taskset = TaskSet(x=np.concatenate((x,mem_x)), y=np.concatenate((y,mem_y)), t=np.concatenate((t,mem_t)), target_trsf=train_taskset.target_trsf, trsf=train_taskset.trsf, data_type=args.tasktype)
    print(f'N samples per class {np.bincount(train_taskset.get_raw_samples()[1])}')
    return prepare_dataloader(args, train_taskset, val_split=0, shuffle=True)[0]

def split_train_val(dataset: BaseTaskSet, val_split: float = 0.1, valid_k: int = 0) -> Tuple[BaseTaskSet, BaseTaskSet]:
    """Split train dataset into two datasets, one for training and one for validation.

    :param dataset: A torch dataset, with .x and .y attributes.
    :param val_split: Percentage to allocate for validation, between [0, 1[.
    :return: A tuple a dataset, respectively for train and validation.
    """
    x,y,t = dataset.get_raw_samples()
    random_state = np.random.RandomState(seed=1)
    
    def selection(x, random_state, val_split):
        indexes = np.arange(len(x))
        random_state.shuffle(indexes)
        val_indexes = indexes[:int(val_split * len(indexes))]
        x_val=x[val_indexes]
        return x_val

    def selection_validk(x, random_state, val_k):
        indexes = np.arange(len(x))
        random_state.shuffle(indexes)
        val_indexes = indexes[:val_k]
        x_val=x[val_indexes]
        return x_val
    unique_classes = np.unique(y)
    if val_split==0 and valid_k>0:
        val_indexes = np.concatenate([selection_validk(np.where(y==c)[0],random_state, int(valid_k)) for c in unique_classes])
    else:
        val_indexes = np.concatenate([selection(np.where(y==c)[0],random_state, val_split) for c in unique_classes])
    train_indexes = np.setdiff1d(np.arange(len(x)),val_indexes)
    # train_x,train_y,train_t = x[train_idx], y[train_idx], t[train_idx]
    # val_x, val_y, val_t = x[val_idx], y[val_idx], t[val_idx]
    
    # indexes = np.arange(len(dataset))
    # random_state.shuffle(indexes)

    # train_indexes = indexes[int(val_split * len(indexes)):]
    # val_indexes = indexes[:int(val_split * len(indexes))]

    # print(train_indexes)
    # print(val_indexes)

    if dataset.data_type != TaskType.H5:
        x_train, y_train, t_train = dataset.get_raw_samples(train_indexes)
        x_val, y_val, t_val = dataset.get_raw_samples(val_indexes)
        idx_train, idx_val = None, None
    else:
        y_train, y_val, t_train, t_val = None, None, None, None
        if dataset._y is not None:
            y_train = dataset._y[train_indexes]
            y_val = dataset._y[val_indexes]

        if dataset._t is not None:
            t_train = dataset._t[train_indexes]
            t_val = dataset._t[val_indexes]
        idx_train = dataset.data_indexes[train_indexes]
        idx_val = dataset.data_indexes[val_indexes]

        x_train = dataset.h5_filename
        x_val = dataset.h5_filename

    train_dataset = TaskSet(x_train, y_train, t_train,
                            trsf=dataset.trsf,
                            target_trsf=dataset.target_trsf,
                            data_type=dataset.data_type,
                            data_indexes=idx_train)
    val_dataset = TaskSet(x_val, y_val, t_val,
                          trsf=dataset.trsf,
                          target_trsf=dataset.target_trsf,
                          data_type=dataset.data_type,
                          data_indexes=idx_val)

    return train_dataset, val_dataset, train_indexes

def plot_test_accs(model, e, task_id, model_name, test_loaders, test_acc_current=None, type_acc:str='test', device=device):
    if test_acc_current is None:
        test_acc_current = test(model, task_id, test_loaders[-1],rank=device)  # exp_state.n_unique_classes[-1], rank=device)  
    accs_previous_tasks = [e,test_acc_current]        
    for t, test_loader in enumerate(test_loaders[:-1]):        
        test_acc_t = test(model,t,test_loader, rank=device)#, exp_state.n_unique_classes[t])
        print('previous task', t, f'{type_acc} acc: ',test_acc_t, 'epoch: ',e,'\n')  
        accs_previous_tasks.append(test_acc_t)
    accs_previous_tasks.append(model_name)
    return accs_previous_tasks

def preprocess_tensor(model:TorchModuleWrapper, x:Tensor, y:Tensor)->Tuple[Tensor, Tensor]:
    device = x.device
    #remap y if needed (for task order permutation of multitask)
    y, _ = remap_class_vector(y, model.mapping)
    # if args.regime=='latent_ER' and args_model.encoder_name not in [None, 'None','fc'] and args_model.classifier_type!='clip_0_shot':
    #     dim = np.sqrt(x.shape[-1])
    #     # while dim % 1!=0:
    #     #     x = x.repeat(0,1)
    #     #     dim = np.sqrt(x.shape[-1])
    #     dim = int(dim)
    #     x= x.view(-1,dim,dim).unsqueeze(1).repeat(1,3,1,1).float() #x.view(-1,dim,dim).unsqueeze(1).repeat(1,3,1,1).float() # n_samples x 3 x dim x dim  
    
    return x.to(device), y.to(device)

def test_nn(model:nn.Module,task_id, test_loader:DataLoader, n_unique_classes:int=None, rank=device, debug=False)-> float:
    model.eval()
    acc = 0
    if len(test_loader)>0:
        with torch.no_grad():
            for i, (x,y,_) in enumerate(tqdm(test_loader)):
                x,y = preprocess_tensor(model,x, y)
                x,y = x.to(rank), y.to(rank)
                if  n_unique_classes is not None:
                    y-=n_unique_classes
                logits = model(x, task_id=task_id) 
                if len(logits.shape)==1: 
                    logits=logits.unsqueeze(0)
                acc += torch.sum(logits.max(1)[1] == y).float()/len(y)   
                if debug and i==5:
                    break   
            acc = acc/len(test_loader)
            acc = acc.cpu().item()
    return acc

def test(model_container:Union[ModelContainer_ER,nn.Module], task_id, test_loader:DataLoader, n_unique_classes:int=None, rank=device, args:ArgsGenerator=None)-> float:
    debug=False
    if args is not None:
        debug=args.debug
        if args.estimate_compute_regime:
            return 0.
    if isinstance(model_container, ModelContainer_ER):
        model = model_container.model
        if model_container.args.classifier_type in ['logistic_regression', 'random_forrest', 'knn']:
            #collect data
            x_test=[]
            y_test=[]
            for x,y,_ in test_loader:
                x,y=preprocess_tensor(model,x,y)
                x_test.append(x)
                y_test.append(y)
            x_test,y_test=torch.cat(x_test).squeeze().cpu().numpy(),torch.cat(y_test).squeeze().cpu().numpy()
            #somehow didnt work with hd5
            # test_features = test_loader.dataset._x
            # test_labels = test_loader.dataset._y 
            if isinstance(model.classifiers, Iterable):
                model_logreg = model.classifiers[task_id]
            else:
                model_logreg = model.classifiers
            predictions = model_logreg.predict(x_test)
            acc = np.mean((y_test == predictions).astype(float)) #* 100.
            return acc
        else:
            return test_nn(model,task_id,test_loader,n_unique_classes, rank=rank, debug=debug)
    else:
        return test_nn(model_container,task_id,test_loader,n_unique_classes, rank=rank, debug=debug)
    
def get_accs_for_tasks(model, loaders:List[DataLoader], accs_past: List[float]=None, device=device, args:ArgsGenerator=None):
    accs=[]        
    Fs = []                             
    for ti, test_loader in enumerate(loaders):     
        acc = test(model,ti,test_loader, rank=device, args=args)#, n_unique_classes=exp_state.n_unique_classes[ti])
        accs.append(acc)
    #####################
        if accs_past is not None:
            Fs.append(acc-accs_past[ti])
    return accs,Fs

class Logger():
    def __init__(self, args:ArgsGenerator, model_container:ModelContainer, nb_tasks) -> None:
        self.nb_tasks=nb_tasks
        self.wandb_F_table= wandb.Table(columns=[i for i in range(nb_tasks)])
        self.wandb_test_acc_table= wandb.Table(columns=[i for i in range(nb_tasks)])
        
        self.metrics=['cosine', 'dot']
        # self.similarity_measurer = SimilarityMeasurer() 
        self.previous_task_protypes =[]
        self.similarity_matrices = {metric:[] for metric in self.metrics}
        self.similarity_means={metric:[] for metric in self.metrics}
        self.task_similarities_table = {metric:wandb.Table(columns=[i for i in range(self.nb_tasks)]) for metric in self.metrics}
        self.best_cvparams_table = wandb.Table(columns=["task", "params_model", "params_classifier"])
        #log model information
        if args.k_shot is not None and args.er_size_per_class>0:
            n_classes=args.n_classes
            if isinstance(n_classes,List):
                n_classes=sum(n_classes)
            log_wandb({'FG_ratio': (args.k_shot*n_classes)/(args.er_size_per_class*n_classes)}, prefix='model')         
        log_wandb({'model/input_size': model_container.args.in_size}, prefix='model')                                   
        log_wandb({'model/n_params(trainable)': sum(p.numel() for p in model_container.model.parameters() if p.requires_grad)}, prefix='model')
        log_wandb({'model/n_params(all)': sum(p.numel() for p in model_container.model.parameters())}, prefix='model')
        if hasattr(model_container.model.classifiers, 'parameters'):
            log_wandb({'model/n_params_classifier': sum(p.numel() for p in model_container.model.classifiers.parameters())}, prefix='model')

    def close(self):        
        for metric in self.metrics:
            wandb.log({f"similarity({metric})": self.task_similarities_table[metric]})        
        wandb.log({"test_forgetting": self.wandb_F_table})
        wandb.log({"test_accuracy": self.wandb_test_acc_table})
        #log table with best param of task level CV
        wandb.log({f"best_params": self.best_cvparams_table})

    # def log_similarity(self, train_taskset, task_id): 
    #     current_task_protypes = self.similarity_measurer.create_prototypes(train_taskset, batch_size=128, merge_type='mean')  
    #     self.previous_task_protypes.append(current_task_protypes)       
    #     for metric in self.metrics:  
    #         similarities_from_current=[0]*self.nb_tasks       
    #         for pt, previous_task_protype in enumerate(self.previous_task_protypes):
    #             similarity_matrix = self.similarity_measurer.calculate_similarity(previous_task_protype, current_task_protypes, metric=metric)
    #             log_wandb({f"similarity/similarity_mean_t{pt}+{task_id}({metric})": np.mean(similarity_matrix)}, prefix='similarity')
    #             self.similarity_means[metric].append(np.mean(similarity_matrix))
    #             log_wandb({f"similarity/similarity_var_t{pt}+{task_id}({metric})": np.var(similarity_matrix)}, prefix='similarity')
    #             self.similarity_matrices[metric].append(similarity_matrix)
    #             similarities_from_current[pt]=np.mean(similarity_matrix)   
    #             log_wandb({f"similarity/mean_task_similarity({metric})":np.mean(self.similarity_means[metric])}, prefix='similarity')
            
    #         self.task_similarities_table[metric].add_data(*similarities_from_current)
    #         # wandb.log({f"similarity({metric})": self.task_similarities_table[metric]})
    
    def log_results(self, args:ArgsGenerator,task_id,model_container,test_loaders_sofar,valid_loaders_sofar,test_accuracies_past,valid_accuracies_past,best_cvparams_table):
        if args.estimate_compute_regime:
            return 
        assert task_id is not None
        #####LOGING####
        replay_buffer_mem=0 
        if model_container.replay_buffer._x is not None:     
            replay_buffer_mem+= model_container.replay_buffer._x.nbytes 
            replay_buffer_mem+= model_container.replay_buffer._y.nbytes 
        log_wandb({'memory(bytes)_ERbuffer': replay_buffer_mem})

        mem_params = sum([param.nelement()*param.element_size() for param in model_container.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model_container.buffers()])
        log_wandb({'memory(bytes)_model': mem_params + mem_bufs})

        #test on all task sofar
        accs_test_table_raw=[0]*self.nb_tasks
        F_test_table_raw=[0]*self.nb_tasks
        accs_test, Fs = get_accs_for_tasks(model_container, test_loaders_sofar, test_accuracies_past, device=args.device, args=args)
        for ti, (acc, Frg) in enumerate(zip(accs_test, Fs)):
            log_wandb({f'test_acc_on_t{ti}':acc}, prefix='test/', step=('task', task_id))
            log_wandb({f'test_acc_on_t{ti}_after_t{task_id}':acc}, prefix='test/', step=('task', task_id))
            accs_test_table_raw[ti]=acc
            #Forgetting (test)
            log_wandb({f'test_F_t{ti}':Frg}, prefix='test/', step=('task', task_id))  
            log_wandb({f'test_F_of_t{ti}_after_t{task_id}':Frg}, prefix='test/', step=('task', task_id))  
            F_test_table_raw[ti]=Frg
            if test_accuracies_past[ti]>0:
                log_wandb({f'test_F_relative{ti}':Frg/test_accuracies_past[ti]}, prefix='test/', step=('task', task_id))
                log_wandb({f'test_F_relative_t{ti}_after_t{task_id}':Frg/test_accuracies_past[ti]}, prefix='test/', step=('task', task_id))
        self.wandb_test_acc_table.add_data(*accs_test_table_raw)
        self.wandb_F_table.add_data(*F_test_table_raw)
        ####################
        #Average accuracy (test) 
        # print(accs_test)
        log_wandb({"mean_test_acc_iid(tasks_sofar)":np.mean(test_accuracies_past)}, step=('task', task_id))#, prefix='test/')
        log_wandb({"mean_test_acc_cl(tasks_sofar)":np.mean(accs_test)}, step=('task', task_id))#, prefix='test/')
        #Average forgetting (test)
        log_wandb({"mean_test_F(tasks_sofar)":np.mean(Fs)}, step=('task', task_id))#, prefix='test/')
        try:
            log_wandb({"mean_test_F_relative(tasks_sofar)":np.mean([f/test_accuracies_past[i] for i,f in enumerate(Fs)])}, step=('task', task_id))#, prefix='test/')
        except:
            pass
        ####################
        #valid on old tasks
        accs_valid, Fs = get_accs_for_tasks(model_container, valid_loaders_sofar, valid_accuracies_past, device=args.device, args=args)
        for ti, (acc, Frg) in enumerate(zip(accs_valid, Fs)):
            log_wandb({f'valid_acc_on_t{ti}':acc}, prefix='valid/', step=('task', task_id))
            log_wandb({f'valid_acc_on_t{ti}_after_t{task_id}':acc}, prefix='valid/', step=('task', task_id))
            #Forgetting (valid)
            log_wandb({f'valid_F_t{ti}':Frg}, prefix='valid/', step=('task', task_id))  
            log_wandb({f'valid_F_of_t{ti}_after_t{task_id}':Frg}, prefix='valid/', step=('task', task_id))  
            if valid_accuracies_past[ti]>0:
                log_wandb({f'valid_F_relative{ti}':Frg/valid_accuracies_past[ti]}, prefix='valid/', step=('task', task_id))
                log_wandb({f'valid_F_relative_t{ti}_after_t{task_id}':Frg/valid_accuracies_past[ti]}, prefix='valid/', step=('task', task_id))
        ####################
        #Average accuracy (valid) 
        # print(accs_test) 
        log_wandb({"mean_valid_acc_iid(tasks_sofar)":np.mean(valid_accuracies_past)}, step=('task', task_id))
        log_wandb({"mean_valid_acc_cl(tasks_sofar)":np.mean(accs_valid)}, step=('task', task_id))
        #Average forgetting (valid)
        log_wandb({"mean_valid_F(tasks_sofar)":np.mean(Fs)}, step=('task', task_id))
        try:
            log_wandb({"mean_valid_F_relative(tasks_sofar)":np.mean([f/valid_accuracies_past[i] for i,f in enumerate(Fs)])}, step=('task', task_id))
        except:
            pass
        ####################
        ####################



#from https://github.com/Continvvm/continuum/blob/class_remap_subscenario/continuum/scenarios/scenario_utils.py
def _get_remapping_classes_ascending_order(new_classes, current_mapping=None):
    """
    Output a vector of classes existing class to get new class labels do:
     new_label = np.where(new_remapping==old_label)[0][0]
    :param new_classes: list of new classes
    :param current_mapping: vector with previous remapping
    """

    array_new_classes = np.array(new_classes)

    if len(np.unique(array_new_classes)) != len(array_new_classes):
        raise ValueError("list new_classes can not contain two time the same class label.")

    ordered_array_new_classes = np.sort(array_new_classes)

    if current_mapping is None:
        new_remapping = ordered_array_new_classes
    else:
        # remove classes already in the mapping
        array_new_classes = np.setdiff1d(array_new_classes, current_mapping)
        if len(array_new_classes) == 0:
            new_remapping = current_mapping
        else:
            new_remapping = np.concatenate([current_mapping, array_new_classes], axis=0)

    return new_remapping

def _remap_class_vector(class_vector, remapping):

    if len(np.where(class_vector == -1)[0]) > 0:
        raise ValueError("-1 is not an acceptable label.")

    if len(np.setdiff1d(class_vector, remapping)) > 0:
        raise ValueError("Some values in class vector are not in the mapping.")

    # we create a new vector to not have interference between old classes and new classes values
    new_vector = np.ones(len(class_vector)) * -1
    for i, key in enumerate(remapping):
        indexes = np.where(class_vector == key)[0]
        new_vector[indexes] = i

    if len(np.where(new_vector == -1)[0]) > 0:
        raise ValueError("Some indexes have not been set in the remapping.")
    return new_vector

def remap_class_vector(class_vector, remapping=None):
    """
    From a mapping vector and a vector of classes output a vector of remapped classes with the mapping eventually updated
    :param class_vector: vector of class labels to remap
    :param remapping: 1D vector with current mapping might be None if the mapping does not exist yet
    """

    unique_classes = np.unique(class_vector)
    if remapping is None or len(np.setdiff1d(unique_classes, remapping)) > 0:
        # here we have some new classes in the vector
        remapping = _get_remapping_classes_ascending_order(new_classes=unique_classes, current_mapping=remapping)

    new_class_vector = _remap_class_vector(class_vector, remapping)
    return torch.tensor(new_class_vector.astype(int)), remapping

def remap_class_vector(class_vector, remapping=None):
    """
    From a mapping vector and a vector of classes output a vector of remapped classes with the mapping eventually updated
    :param class_vector: vector of class labels to remap
    :param remapping: 1D vector with current mapping might be None if the mapping does not exist yet
    """

    unique_classes = np.unique(class_vector)
    if remapping is None or len(np.setdiff1d(unique_classes, remapping)) > 0:
        # here we have some new classes in the vector
        remapping = _get_remapping_classes_ascending_order(new_classes=unique_classes, current_mapping=remapping)

    new_class_vector = _remap_class_vector(class_vector, remapping)
    return torch.tensor(new_class_vector.astype(int)), remapping

def get_scenario_remapping(scenario):
    mapping = None
    for taskset in scenario:
        unique_classes = taskset.get_classes()
        _, mapping = remap_class_vector(unique_classes, mapping)
    return mapping

def split_er_buffer(buffer:RehearsalMemory, val_split:float, data_type)-> Tuple[TaskSet, TaskSet]:
    x,y,t = buffer.get()
    random_state = np.random.RandomState(seed=1)
    
    def selection(x, random_state, val_split):
        indexes = np.arange(len(x))
        random_state.shuffle(indexes)
        # train_indexes = indexes[int(val_split * len(indexes)):]
        val_indexes = indexes[:int(val_split * len(indexes))]
        # x_train=x[train_indexes]
        x_val=x[val_indexes]
        return x_val
    unique_classes = np.unique(y)
    val_idx = np.concatenate([selection(np.where(y==c)[0],random_state, val_split) for c in unique_classes])
    train_idx = np.setdiff1d(np.arange(len(x)),val_idx)
    train_x,train_y,train_t = x[train_idx], y[train_idx], t[train_idx]
    val_x, val_y, val_t = x[val_idx], y[val_idx], t[val_idx]
    
    train_er_buffer = TaskSet(train_x, train_y, train_t,
                            trsf=None,
                            data_type=data_type)
    val_er_buffer = TaskSet(val_x, val_y, val_t,
                          trsf=None,
                          data_type=data_type)
    return train_er_buffer, val_er_buffer

def create_subscenario(base_scenario, task_indexes):
    """
    In this function we want to create a subscenario from the different tasks, either by subsampling tasks or reodering
    or both.
    """

    if torch.is_tensor(task_indexes):
        task_indexes = task_indexes.numpy()

    if base_scenario.transformations is not None and isinstance(base_scenario.transformations[0], list):
        transformations = [base_scenario.transformations[i] for i in task_indexes]
    else:
        transformations = base_scenario.transformations
    sub_scenario = None
    if isinstance(base_scenario, OnlineFellowship):
        # We just want to changes base_scenario.cl_datasets order
        new_cl_datasets = [base_scenario.cl_datasets[i] for i in task_indexes]
        sub_scenario = OnlineFellowship(new_cl_datasets,
                                        transformations=transformations,
                                        update_labels=base_scenario.update_labels)
    elif base_scenario.cl_dataset.data_type == TaskType.H5:
        list_taskset = [base_scenario[i] for i in task_indexes]
        sub_scenario = OnlineFellowship(list_taskset,
                                        transformations=transformations,
                                        update_labels=False)
    else:
        new_x, new_y, new_t = None, None, None
        if base_scenario.cl_dataset.bounding_boxes is not None:
            raise ValueError("the function create_subscenario is not compatible with scenario with bounding_boxes yet.")

        for i, index in enumerate(task_indexes):
            taskset = base_scenario[index]
            all_task_indexes = np.arange(len(taskset))
            x, y, t = taskset.get_raw_samples(all_task_indexes)
            t = np.ones(len(y)) * i
            if new_x is None:
                new_x = x
                new_y = y
                new_t = t
            else:
                new_x = np.concatenate([new_x, x], axis=0)
                new_y = np.concatenate([new_y, y], axis=0)
                new_t = np.concatenate([new_t, t], axis=0)
        dataset = InMemoryDataset(new_x, new_y, new_t, data_type=base_scenario.cl_dataset.data_type)
        sub_scenario = ContinualScenario(dataset, transformations=transformations)

    return sub_scenario

class SumMeter(AverageValueMeter):
    def __init__(self) -> None:
        super().__init__()
        self.sum =0.
    def value(self):
        return self.sum, None

def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i
     
def get_schedule(dataset_size):
  if dataset_size < 20_000:
    return [100, 200, 300, 400, 500]
  elif dataset_size < 500_000:
    return [500, 3000, 6000, 9000, 10_000]
  else:
    return [500, 6000, 12_000, 18_000, 20_000]


def is_connected(host='http://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False

def get_lr(step, dataset_size, base_lr=0.003):
  """Returns learning-rate for `step` or None at the end."""
  supports = get_schedule(dataset_size)
  # Linear warmup
  if step < supports[0]:
    return base_lr * step / supports[0]
  # End of training
  elif step >= supports[-1]:
    return None
  # Staircase decays by factor of 10
  else:
    for s in supports[1:]:
      if s < step:
        base_lr /= 10
    return base_lr

known_dataset_sizes = {
  'cifar10': (32, 32),
  'cifar100': (32, 32),
  'oxford_iiit_pet': (224, 224),
  'oxford_flowers102': (224, 224),
  'imagenet2012': (224, 224),
}

def get_resolution(original_resolution):
  """Takes (H,W) and returns (precrop, crop)."""
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)


def get_resolution_from_dataset(dataset):
  if dataset not in known_dataset_sizes:
    raise ValueError(f"Unsupported dataset {dataset}. Add your own here :)")
  return get_resolution(known_dataset_sizes[dataset])


def bn_eval(module:torch.nn.Module, freeze=True):
    for layer in module.children():      
        if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.BatchNorm3d):
          if freeze:
            layer.eval()
          else:
            layer.train()
        elif isinstance(layer.children(), Iterable):
            bn_eval(layer, freeze)

def cleanup(message: Dict[str, Union[Dict, str, float, Any]], sep: str="/") -> Dict[str, Union[str, float, Any]]:
    """Cleanup a message dict before it is logged to wandb.

    Args:
        message (Dict[str, Union[Dict, str, float, Any]]): [description]
        sep (str, optional): [description]. Defaults to "/".

    Returns:
        Dict[str, Union[str, float, Any]]: [description]
    """
    # Flatten the log dictionary
    
    message = flatten_dict(message, separator=sep)

    # TODO: Remove redondant/useless keys
    for k in list(message.keys()):
        if k.endswith((f"{sep}n_samples", f"{sep}name")):
            message.pop(k)
            continue

        v = message.pop(k)
        # Example input:
        # "Task_losses/Task1/losses/Test/losses/rotate/losses/270/metrics/270/accuracy"
        # Simplify the key, by getting rid of all the '/losses/' and '/metrics/' etc.
        things_to_remove: List[str] = [f"{sep}losses{sep}", f"{sep}metrics{sep}"]
        for thing in things_to_remove:
            while thing in k:
                k = k.replace(thing, sep)
        # --> "Task_losses/Task1/Test/rotate/270/270/accuracy"
        if 'Task_losses' in k and 'accuracy' in k and not 'AUC' in k:
            k = k.replace('Task_losses', 'Task_accuracies')

        if 'Cumulative' in k and 'accuracy' in k and not 'AUC' in k:
            k = 'Task_accuracies/'+k
        
        if 'coefficient' in k:
            k = 'coefficients/'+k
        
        # Get rid of repetitive modifiers (ex: "/270/270" above)
        parts = k.split(sep)
        k = sep.join(unique_consecutive(parts))
        # Will become:
        # "Task_losses/Task1/Test/rotate/270/accuracy"
        
        if isinstance(v, Iterable):
            for i, el in enumerate(v):
                k_new = k + f'/{i}'
                message[k_new] = el
        else:
            message[k] = v

    return message

def add_prefix(some_dict: Dict[str, T], prefix: str="") -> Dict[str, T]:
    """Adds the given prefix to all the keys in the dictionary that don't already start with it. 
    
    Parameters
    ----------
    - some_dict : Dict[str, T]
    
        Some dictionary.
    - prefix : str, optional, by default ""
    
        A string prefix to append.
    
    Returns
    -------
    Dict[str, T]
        A new dictionary where all keys start with the prefix.
    """
    if not prefix:
        return OrderedDict(some_dict.items())
    result: Dict[str, T] = OrderedDict()
    for key, value in some_dict.items():
        new_key = key if key.startswith(prefix) else (prefix + key)
        result[new_key] = value
    return result

def log_wandb(message, step:Tuple[str,Any]=None, prefix=None, print_message=False, clean=True):
    # for k, v in message.items():
    #         if hasattr(v, 'to_log_dict'):
    #             message[k] = v.to_log_dict()
    if clean:
        try: 
            message = cleanup(message, sep="/")
        except:
            pass
    if prefix:
        message = add_prefix(message, prefix)
    if step is not None:
        message[step[0]] = step[1]
    try:
        wandb.log(message)#, step=step)
    except: # ValueError:
        pass #wandb is not innitialized
    if print_message:
        print(message)

def set_seed(manualSeed=None):
    assert manualSeed is not None
    #####seed#####
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    if device != "cpu":
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    ######################################################


D = TypeVar("D", bound=Dict)
def flatten_dict(d: D, separator: str="/") -> D:
    """Flattens the given nested dict, adding `separator` between keys at different nesting levels.

    Args:
        d (Dict): A nested dictionary
        separator (str, optional): Separator to use. Defaults to "/".

    Returns:
        Dict: A flattened dictionary.
    """
    result = type(d)()
    for k, v in d.items():
        if isinstance(v, dict):
            for ki, vi in flatten_dict(v, separator=separator).items():
                key = f"{k}{separator}{ki}"
                result[key] = vi
        else:
            result[k] = v
    return result



def unique_consecutive(iterable: Iterable[T], key: Callable[[T], Any]=None) -> Iterable[T]:
    """List unique elements, preserving order. Remember only the element just seen.
    
    >>> list(unique_consecutive('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> list(unique_consecutive('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']
    
    Recipe taken from itertools docs: https://docs.python.org/3/library/itertools.html
    """

    import operator  
    from itertools import groupby
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))
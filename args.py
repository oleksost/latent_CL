

import os 
import torch
import hashlib
import copy
import numpy as np      
from dataclasses_json import dataclass_json
from continuum.tasks.base import TaskType  
from dataclasses import dataclass
from simple_parsing import choice,field
from Models.helper import getKeysByValue
from typing import Callable, Iterable, NamedTuple, Optional, Tuple, Union, List
from Models.encoders import encoders
from Data.datasets import datasets as dataset_list,dataset_tuple
  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
task_orderings={
    # random orderings, generated onse and shared across runs
    '4': [[0,1,2,3], [2,3,0,1], [3,2,1,0], [3,0,1,2]], #[1,2,3,4,5],
    '5': [[0,1,2,3,4], [2,3,4,0,1], [3,2,1,4,0], [3,4,0,1,2], [2,3,4,1,0], [0,3,4,2,1]], #[1,2,3,4,5],
    '6': [[0,1,2,3,4,5], [2,3,4,0,1,5], [5,3,2,1,4,0], [3,4,0,1,2,5], [2,3,4,5,1,0], [0,5,3,4,2,1]],
}

@dataclass_json
@dataclass#(eq=True, frozen=False)
class ArgsGenerator(): 
    stop_after_first_task: bool = 0 # Useful for hp_tuning with task level tuning if need to select hp on the first task
    seed: int = 180 #-

    #data strean   
    permute_task_order: bool = 0 # if 'True' task order is permuted conditioned on the seed
    n_classes: Optional[int] = None #-
    dataset_name: str = choice(dataset_list.keys(), default='CIFAR10')
    n_tasks: int = 2 #-
    k_shot:Optional[int] = None #number of shots per class in each new task;
    n_task_order_permutations: int = 0 # if >0 runs n_task_order_permutations runs with different task order

    #optimization
    epochs: int = 100 #-
    batch_size: int = 64 #-

    #encoding
    concat_dataset_encoders_hidden_n: Optional[str] = None #-
    concat_dataset_encoders: List[str]=field(default_factory=list) # path a list of feature encoders and their features will be concatenated
    regenerate_encodings: bool = 0 #-
    encoding_batch_size: int = 64
    dataset_encoder_name:str = choice(encoders.keys(), default='ViT-B/32')# used for dataset encoding in case of latent_ER
    encode_with_continuum: bool = 1 #-

    #er buffer       
    balancing_strategy: str = choice('oversample', 'loss', default='oversample')
    size_of_random_prototype_factor: Optional[float] = None #if not None, the number of samples from which the prototypes will be created is n = int(n_samples_per_class / er_size_per_class), 
                                                            #then n is multiplied by size_of_random_prototype_factor, samples fro ptototype cretion are samples with replacement in this case. 
    
    #TODO: is er_with_prototypes an invariant attribute?
    er_with_prototypes: int = 0 # if 'True', each ample in the replay butter if the prototype of the proportional part of the training data of that class 
    er_buffer_type:str = choice('balanced', 'random', 'balanced_batch', default='random')
    er_size_per_class: int = 0 #-
     
    resolution: Optional[int] = None # Imposes resolution on dataset encodings, if None uses standard resolutions defined in datasets.py

    #regime          
    #finetuning        
    finetuning_only_norms: bool = 0 #-
    unfreeze_input_layer: bool = 0 #-

    use_predefined_orderings: bool = 0 # if 'True' use orderings from task_orderings
    task_order_id: Optional[int] = None #- 
    simulate_iid: bool = 0 # if 'True' simulates the iid training of all tasks seen sofar
    concat_test_sets: bool = 0 # if 'True' current task's test set is a concatenetion of all rpevious tasks' test sets
    concat_validation_sets: bool = 0 # if 'True' current task's validation set is a concatenetion of all rpevious task's validation sets
    reinit_between_tasks: bool = 0 # -
    device: str = choice('cuda', 'cpu', default='cuda')
    estimate_compute_regime: bool = 0 #-
    estimate_time:bool = False #
    estimate_compute_regime_encoding:bool = 0 #-
    debug: bool = 0 #-
    regime: int = choice('latent_ER', 'sample_ER', default='latent_ER') #-
    schedule: str = choice('normal', 'classifier+normal', default='normal') # classifier+normal --- first trains only classifier untill convergence, then finetunes (unfreezes) representation
    early_stopping:bool = 0 # -
    early_stopping_patience: int = 10 # patience parameter in epochs for early stopping

    #reporting       
    wandb_entity: Optional[str] = None #
    wandb_project:str = 'large_cl' #-
    wandb_notes: str = '' #-
    group_name: Optional[str] = None #-
    test_every: int = 5 #-  
    record_flops: bool = 1#-
    validate_every: int = 1 #-
    log_task_similarity: bool = 0 #-
    save_final_model: bool = 1 #-    
    test_old_tasks: bool = 0 #- if 'True' logs a wandb table of accuracies on all tasks seen sofar every 'test_every' epochs (can be used to reproduce the figure from Ethan Dyer's video)

    #cross validation
    valid_fraction: float = 0.1 #fraction of the training data used for validation
    valid_k: Optional[float]=0 # number of smaples to use for validation (if valid_fraction>0 it is ignored)
    research_hp:bool = 0 #if 1 would remove existing best hps in hp_database and do the search again
    use_hp_database: bool = 0 # it 'True' stores and loads hyperparameter in/to a Tiny database
    task_level_cv: bool = 0 # if 'True', does task-level cross validation, i.e. optimal parameters are selected for each task seperately as opposed to the whole tasks stream
    keep_best_params_after_first_task: bool = 0# if 'True' and task_level_cv, keeps optimal parameters selected on the task 1 for the rest of the stream
    fraction_buffer_samples_valid: float = 0. # the fraction os samples per class fromt he replay buffer to add to the cvalidation set of the current task
    n_epochs_task_level_cv: Optional[int] = None # number of epochs for per task hyperparameter tuning 
    
    #set automatically
    data_path: Optional[str] = None # path where to store data and encodings
    weights_path: Optional[str] = None # path where to store model weights
    class_order:Optional[List] = None
    er_size_total:Optional[int] = None #-
    ddp:bool = False #-
    num_workers: int = 0 #number dataloading workers
    tasktype: Optional[int] = None #-
    dataset:Optional[dataset_tuple] = None
    dataset_encoders:str = '' #- 
    hp_db_path: Optional[str] = None # path where to store hyperparameter dataset
    task_order: Optional[List] = None

    #encoding     
    freeze_vit_untill_layer: Optional[int]= None #-
    learn_bn_stats_before_encode: Optional[int] = 0 #

    def get_hp_db_path(self):
        project_home = f"{os.environ.get('LARGE_CL_HOME')}" if "LARGE_CL_HOME" in os.environ else "."
        #on mila cluster copy datasets to $SLURM_TMPDIR

        if self.hp_db_path is None: 
            hp_db_path=f'{project_home}'
        else:
            hp_db_path=self.hp_db_path
        if not os.path.exists(hp_db_path):
            os.makedirs(hp_db_path)
        return hp_db_path

    def get_data_path(self):
        project_home = f"{os.environ.get('LARGE_CL_HOME')}" if "LARGE_CL_HOME" in os.environ else "."
        #on mila cluster copy datasets to $SLURM_TMPDIR

        if self.data_path is None: 
            # if os.path.isdir(os.path.join(os.environ['HOME'],"scratch/large_CL/Datasets")):
            #     data_path=f"{os.environ['SCRATCH']}/large_CL/Datasets" #CUB_200_2011/"
            # else:
            data_path=f'{project_home}/Data'
        else:
            data_path=self.data_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if self.dataset_name=='DomainNet':
            data_path+='/DomainNet'
        return data_path

    def get_weights_path(self):
        if self.weights_path is None:
            project_home = f"{os.environ.get('LARGE_CL_HOME')}" if "LARGE_CL_HOME" in os.environ else "."
            models_path = f"{os.environ.get('SCRATCH')}/Weights/" if "SCRATCH" in os.environ else f"{project_home}/Weights/"
            weights_path= models_path
        else:
            weights_path=self.weights_path
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        return weights_path
    
    def generate_task_order(self, run=0):
        if self.permute_task_order:
            if self.task_order_id is not None and self.n_task_order_permutations==1:
                run = self.task_order_id
            if self.use_predefined_orderings:
                self.task_order = task_orderings[f'{self.n_tasks}'][run]
            else:  
                self.task_order = np.random.permutation(self.n_tasks)     
        else:
            self.task_order = np.arange(self.n_tasks)

    def __post_init__(self):
        if isinstance(self.concat_dataset_encoders_hidden_n, str):
            self.concat_dataset_encoders_hidden_n = [int(n) for n in self.concat_dataset_encoders_hidden_n.split(',')]
        if self.device=='cuda':
            self.device = device
        if self.dataset_name=='DomainNet':
            self.n_tasks=6

        self.data_path = self.get_data_path()
        self.hp_db_path = self.get_hp_db_path()
        self.weights_path = self.get_weights_path() 
        self.dataset = dataset_list[self.dataset_name]
        self.dataset = self.dataset._replace(dataset_info=self.dataset.dataset_info(n_tasks=self.n_tasks, resolution=self.resolution, data_path=self.data_path))
        self.dataset_increments = self.dataset.dataset_info.increment
        self.generate_task_order()

        if self.simulate_iid:    
            self.er_size_per_class = 10000000 #make sure we fit all classes in the er memory
            self.er_buffer_type = 'random'
            self.concat_validation_sets = 1
            self.reinit_between_tasks = 1
        self.er_size_total = self.er_size_per_class * self.dataset.dataset_info.n_classes

        if len(self.concat_dataset_encoders)>1:
            self.dataset_encoder_name='multiple'
            self.dataset_encoders=str(self.concat_dataset_encoders)
        if self.regime=='sample_ER':
            self.tasktype=TaskType.IMAGE_ARRAY if self.dataset.dataset_info.tasktype is None else self.dataset.dataset_info.tasktype 
        else:
            self.tasktype=TaskType.TENSOR
        
        self.n_tasks=self.dataset.dataset_info.n_tasks
        self.n_classes=self.dataset.dataset_info.n_classes_per_task
        if not self.encode_with_continuum:
            raise NotImplementedError
        if self.debug:
            self.epochs = 3 #min(10, self.epochs)

        if self.group_name is None:     
            self.group_name = self.generate_group_name()
    
    def generate_group_name(self, prefix=''):
        return prefix+self.md5+str(self.er_size_per_class)

    @property
    def md5(self): 
        self_copy = copy.copy(self) 
        invariant_args = self_copy.get_hp_invariant_attributes()
        for arg in invariant_args.keys():
            # setattr(self_copy,arg,None)
            delattr(self_copy,arg)
        self_copy.dataset=None        
        return hashlib.md5(str(self_copy).encode('utf-8')).hexdigest()
    
    def serializable_copy(self):
        self_copy = copy.copy(self)
        #remove not serializable attributes like partial functions
        self.class_order=None
        self_copy.task_order=None
        self_copy.dataset=None
        self_copy.tasktype=None
        return self_copy

    def get_hp_invariant_attributes(self):
        '''
        should return attributes which do not influence hyperparameter tuning
        '''
        attributes = ['data_path','weights_path','hp_db_path', 'er_with_prototypes', 'research_hp', 'use_predefined_orderings', 'task_order_id', 'seed','task_order','n_task_order_permutations', 'reinit_between_tasks', 'group_name','wandb_project', 'wandb_notes', 'test_every','regenerate_encodings', 'encoding_batch_size', 'stop_after_first_task', 'log_task_similarity', 'validate_every', 'record_flops']
        if self.keep_best_params_after_first_task:
            attributes+=['er_size_per_class','fraction_buffer_samples_valid','er_buffer_type', 'er_size_total']
        self_dict = self.to_dict()
        return {key: self_dict[key] for key in attributes}

@dataclass(frozen=False)
class ExperimentState: 
    current_task: int = 0            
    n_unique_classes: List[int] = field(default_factory=list)
    def __post_init__(self):
        self.n_unique_classes.append(0)


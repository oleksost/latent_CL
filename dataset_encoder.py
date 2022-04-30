import copy
import os
import time
import wandb
import torch
import shutil
import numpy as np   
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List, Union
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F
from simple_parsing import ArgumentParser, choice
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose, Lambda, ToTensor

import continuum
from args import ArgsGenerator 
from continuum import ClassIncremental
from continuum.datasets import _ContinuumDataset 
from continuum.datasets import H5Dataset, InMemoryDataset
from continuum.tasks import TaskType
from Models.helper import getKeysByValue
from continuum.tasks.task_set import TaskSet
from continuum import TransformationIncremental
from Models.model import ModelContainer   
from continuum.scenarios import ContinualScenario, InstanceIncremental
from Utils.utils import SumMeter, bn_eval, is_connected, log_wandb, set_seed
from continuum.scenarios import OnlineFellowship, encode_scenario, create_subscenario
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class UsedFlops:
    total_gigaflops_fw: float
    total_gigaflops_bw: float
    total_gigaflops: float
    total_gigaflops_fw_es: float
    total_gigaflops_bw_es: float
    total_gigaflops_es: float
    walltime_task: float
    total_flops_fw_epoch: float
    total_flops_bw_epoch: float
    walltime_epoch: float

def select_fewshot_subset_per_class(base_scenario, k_shot):
    """
    In this function we want to create a scenario with k samples per class, where samples are chosen randomly.
    """

    if base_scenario.transformations is not None and isinstance(base_scenario.transformations[0], list):
        transformations = [base_scenario.transformations[i] for i in range(base_scenario.nb_tasks)]
    else:
        transformations = base_scenario.transformations

    def selection(x, random_state, k_shot):
        indexes = np.arange(len(x))
        random_state.shuffle(indexes)
        val_indexes = indexes[:k_shot]
        # x_train=x[train_indexes]
        x_val=x[val_indexes]
        return x_val
    
    sub_scenario = None    
    new_x, new_y, new_t = None, None, None
    if base_scenario.cl_dataset.bounding_boxes is not None:
        raise ValueError("Not compatible with scenario with bounding_boxes yet.")

    for i, index in enumerate(range(base_scenario.nb_tasks)):
        taskset = base_scenario[index]
        # all_task_indexes = np.arange(len(taskset))
        # x, y, t = taskset.get_raw_samples(all_task_indexes)
        y = taskset._y #get_classes()
        unique_classes = np.unique(y)
        random_state = np.random.RandomState(seed=1)
        idxs = np.concatenate([selection(np.where(y==c)[0],random_state, k_shot) for c in unique_classes])
        x, y, t = taskset.get_raw_samples(np.sort(idxs))
        # x,y,t = x[idxs], y[idxs], t[idxs]
        if isinstance(base_scenario, OnlineFellowship):
            t = np.repeat(i, len(t))
        if new_x is None:
            new_x = x
            new_y = y
            new_t = t
        else:
            new_x = np.concatenate([new_x, x], axis=0)
            new_y = np.concatenate([new_y, y], axis=0)
            new_t = np.concatenate([new_t, t], axis=0)
    dataset = InMemoryDataset(new_x, new_y, new_t, data_type=TaskType.TENSOR)#base_scenario.cl_dataset.data_type)
    sub_scenario = ContinualScenario(dataset, transformations=transformations)
    return sub_scenario
                       
def concat_datasets_featurewise(datasets:List[_ContinuumDataset])->_ContinuumDataset:
    '''
    This function concatenates features from datasets along the 1st dimention
    '''
    assert int(sum([len(ds) for ds in datasets])/len(datasets))==len(datasets[0])
    xs, ys, ts = [], None, None
    for dataset in datasets:  
        scenario = ClassIncremental(dataset, nb_tasks=1)
        x, y, t = scenario[0].get_raw_samples()
        if ys is None:
            ys = y
        if ts is None:
            ts = t
        xs.append(x)
    xs = np.concatenate(xs, axis=1)
    # assert xs.shape[-1]==xs[0]*len(datasets)
    dataset = InMemoryDataset(xs, y, t, data_type=TaskType.TENSOR)
    return dataset, xs.shape[-1]
    
def _check_dataset(args:ArgsGenerator, dataset):
    x,y,t = dataset.get_data()
    if min(y)>0:
        y-=min(y)         
        return InMemoryDataset(x,y,t,data_type=dataset.data_type)
    else:
        return dataset

def get_encoded_datasets(args:ArgsGenerator,args_model:ModelContainer.Options, in_size, regenerate, encoding_name_sufix=''): 
    """
    Encode dataset with a single encoder
    """
    # deal with possibly using models from different checkpoints, encoding with different hidden layer etc.
    if args.resolution is not None:
        encoding_name_sufix+=f'_resolution{args.resolution}'
    if args_model.hidden_layer_n is not None:
        # assert args_model.encoder_name.isin(['ViT-B/16','ViT-B/32'])
        encoding_name_sufix+=f'_hidden_layer{args_model.hidden_layer_n}'
    if args_model.checkpoint is not None:
        encoding_name_sufix+=f'_checkpoint_{args_model.checkpoint}'
    encoder = DatasetEncoderContinuum(args.dataset_encoder_name, args_model, args, input_size=in_size, name_sufix=encoding_name_sufix)  
    args_model.in_size = int(encoder.feature_size)
    scenario, scenario_test = prepare_scenarios_sample(args,args_model,transformations=encoder.transforms,transforms_val=encoder.transforms_val, n_tasks=1, increment=0, dataset_name=args.dataset_name)         
    dataset_train, dataset_test = encoder.load_scenarios(scenario=scenario, scenario_test=scenario_test, data_path=args.data_path, regenerate=regenerate)
    return dataset_train, dataset_test, encoder
      
def prepare_scenarios_latent(args:ArgsGenerator, args_model:ModelContainer.Options, regenerate, n_tasks, increment):
    """
    Prepares encoded scenarios or loads them from disc.
    Deals with "Big" stream (stream contaning os several datasets)
    Deals mixture of encoders (concatenating fetures encoded by different encoders).
    """
    in_size=None                                     
    if len(args.dataset.dataset_info.list_datasets)==0:  
        if len(args.concat_dataset_encoders)>1:                
            #concatenate features from different encoders: mixture of encoders
            datasets_train, datasets_test, name = [], [], ''
            for enc_i, encoder_name in enumerate(args.concat_dataset_encoders):
                args_copy=copy.copy(args)
                args_model_copy=copy.copy(args_model)
                if args.concat_dataset_encoders_hidden_n is not None:
                    hidden_n=args.concat_dataset_encoders_hidden_n[enc_i]
                    args_model_copy.hidden_layer_n=hidden_n
                args_copy.dataset_encoder_name=encoder_name
                dataset_train, dataset_test, encoder = get_encoded_datasets(args_copy, args_model_copy, in_size, regenerate)
                name+=encoder.train_data_file.replace('.hdf5','')
                datasets_train.append(dataset_train)
                datasets_test.append(dataset_test)
            del encoder
            # name_train = name
            # name_test = name_train.replace('train', 'test')
            dataset_train, feature_size=concat_datasets_featurewise(datasets_train)
            dataset_test, _ = concat_datasets_featurewise(datasets_test)
            args_model.in_size=int(feature_size)
        else:
            dataset_train, dataset_test, encoder = get_encoded_datasets(args, args_model, in_size, regenerate)
            del encoder
        if args.dataset.dataset_info.increment!=0:
            n_tasks=0
        transformations=[Lambda(lambda x: x)] #doesnt do anything
        if args.dataset_name=='DomainNet':
            scenario = InstanceIncremental(dataset_train, transformations=transformations, nb_tasks=n_tasks)
            scenario_test = InstanceIncremental(dataset_test, transformations=transformations, nb_tasks=n_tasks)
        elif  args.dataset_name=='MNIST_bckgrndwap':
            scenario = TransformationIncremental(dataset_train, base_transformations=[ToTensor()], incremental_transformations=transformations) 
            scenario_test = TransformationIncremental(dataset_test, base_transformations=[ToTensor()], incremental_transformations=transformations)
        else:
            scenario = ClassIncremental(dataset_train, class_order=args.class_order, increment=increment, transformations=transformations, nb_tasks=n_tasks)
            scenario_test = ClassIncremental(dataset_test, class_order=args.class_order, increment=increment, transformations=transformations, nb_tasks=n_tasks)
    else:
        # Big scenario: learning different datasets incrementally
        # use OnlineFellowship
        encoded_datasets_train=[]
        encoded_datasets_test=[]      
        n_classes_sofar=0 
        
        for dn, dataset in enumerate(args.dataset.dataset_info.list_datasets):  
            if len(args.concat_dataset_encoders)>1:                
                #concatenate features from different encoders
                datasets_train, datasets_test, name = [], [], ''
                for encoder_name in args.concat_dataset_encoders:
                    args_copy=copy.copy(args)
                    args_copy.dataset=dataset
                    args_copy.dataset_name=dataset.dataset_info.name
                    args_copy.dataset_encoder_name=encoder_name
                    dataset_train, dataset_test, encoder = get_encoded_datasets(args_copy, args_model, in_size, regenerate)
                    name+=encoder.train_data_file.replace('.hdf5','')
                    datasets_train.append(dataset_train)
                    datasets_test.append(dataset_test)
                    del encoder
                # name_train = name
                # name_test = name_train.replace('train', 'test')
                dataset_train, feature_size=concat_datasets_featurewise(datasets_train)
                dataset_test, _ = concat_datasets_featurewise(datasets_test)

                args_model.in_size=int(feature_size)
            else:
                args_copy=copy.copy(args)
                args_copy.dataset=dataset
                args_copy.dataset_name=dataset.dataset_info.name
                encoding_name_sufix=''
                if args.dataset_name=='BigAugm':
                    # regenerate=True
                    encoding_name_sufix=f'_BigAugm_task{dn}'
                    args_copy.dataset.dataset_info.transformations = args.dataset.dataset_info.transformations[dn]
                    args_copy.dataset.dataset_info.transformations_val = args.dataset.dataset_info.transformations_val[dn]
                dataset_train, dataset_test, encoder = get_encoded_datasets(args_copy, args_model, in_size, regenerate, encoding_name_sufix=encoding_name_sufix)
                del encoder       
            ####################
            # Make sure that task labels are in the correct range 
            # (alternatively set update_labels=False in OnlineFellowship, however since it would create a target transform, it doesnt currently work with replay buffer)           
            x,y,t = dataset_train.get_data()
            y+=n_classes_sofar
            dataset_train = TaskSet(x,y,t,trsf=dataset_train.transformations,data_type=dataset_train.data_type )

            x,y,t = dataset_test.get_data()
            y+=n_classes_sofar
            n_classes_sofar+=len(np.unique(y)) 
            dataset_test = TaskSet(x,y,t,trsf=dataset_test.transformations,data_type=dataset_train.data_type)
            ####################
            encoded_datasets_train.append(dataset_train)
            encoded_datasets_test.append(dataset_test)
                
        if args.dataset.dataset_info.increment!=0:
            n_tasks=0

        transformations=[Lambda(lambda x: x)] #doesnt do anything
        scenario = OnlineFellowship(encoded_datasets_train, update_labels=False, transformations=transformations)
        scenario_test = OnlineFellowship(encoded_datasets_test, update_labels=False, transformations=transformations)   
    return scenario, scenario_test

def prepare_scenarios_sample(args:ArgsGenerator, args_model:ModelContainer.Options, transformations:Union[Callable, List], transforms_val:Union[Callable, List], n_tasks, increment, dataset_name):
    if dataset_name in ['Big', 'BigAugm']:     
        dataset_train_list, dataset_test_list = args.dataset.get_datasets(args.dataset.dataset_info.list_datasets,args.data_path)    
        n_classes_sofar=0
        for i, (dataset_train,  dataset_test) in enumerate(zip(dataset_train_list, dataset_test_list)):
            ####################
            # Make sure that task labels are in the correct range 
            # (alternatively set update_labels=False in OnlineFellowship, however since it would create a target transform, it doesnt currently work with replay buffer)           
            x,y,t = dataset_train.get_data()
            y+=n_classes_sofar
            dataset_train = TaskSet(x,y,t,trsf=Compose(transformations), data_type=dataset_train.data_type)
            dataset_train_list[i]=dataset_train

            x,y,t = dataset_test.get_data()
            y+=n_classes_sofar
            n_classes_sofar+=len(np.unique(y))   
            dataset_test = TaskSet(x,y,t,trsf=Compose(transformations), data_type=dataset_train.data_type)
            dataset_test_list[i]=dataset_test
            ####################

        scenario = OnlineFellowship(dataset_train_list, update_labels=False, transformations=transformations)
        scenario_test = OnlineFellowship(dataset_test_list, update_labels=False, transformations=transformations)
    else:
        dataset_train, dataset_test = args.dataset.get_datasets(args.data_path)           
        dataset_train,dataset_test = _check_dataset(args,dataset_train), _check_dataset(args,dataset_test)   
        x,y,t = dataset_train.get_data()
        if args.dataset_name=='DomainNet':
            scenario = InstanceIncremental(dataset_train, transformations=transformations, nb_tasks=n_tasks)
            scenario_test = InstanceIncremental(dataset_test, transformations=transformations, nb_tasks=n_tasks)  
        elif  args.dataset_name=='MNIST_bckgrndwap':
            scenario = TransformationIncremental(dataset_train, base_transformations=[ToTensor()], incremental_transformations=transformations) 
            scenario_test = TransformationIncremental(dataset_test, base_transformations=[ToTensor()], incremental_transformations=transforms_val)
        else:
            scenario = ClassIncremental(dataset_train, class_order=args.class_order, increment=increment, transformations=transformations, nb_tasks=n_tasks)
            scenario_test = ClassIncremental(dataset_test, class_order=args.class_order, increment=increment, transformations=transforms_val, nb_tasks=n_tasks)
        
        if args.dataset.dataset_info.append_transfrom is not None and not 'clip' in args.dataset_encoder_name:
            scenario.transformations.append(args.dataset.dataset_info.append_transfrom)  
            scenario_test.transformations.append(args.dataset.dataset_info.append_transfrom)
    return scenario, scenario_test

def prepare_scenarios(args:ArgsGenerator, args_model:ModelContainer.Options, transformations:Union[Callable, List]=None, transforms_val:Union[Callable, List]=None, regenerate=None)-> continuum.scenarios._BaseScenario:
    '''
    Prepares CL scenario.
    '''
    if regenerate is None:
        regenerate=args.regenerate_encodings

    n_tasks=args.n_tasks 
    increment=args.dataset.dataset_info.increment
    if increment!=0:
        n_tasks=0
    k_shot=args.k_shot
    regime =args.regime    
    dataset_name = args.dataset_name

    if regime=='latent_ER': 
        scenario, scenario_test = prepare_scenarios_latent(args, args_model, regenerate, n_tasks, increment)
        
    elif regime=='sample_ER':
        scenario, scenario_test = prepare_scenarios_sample(args, args_model, transformations, transforms_val, n_tasks, increment, dataset_name)
    else:
        raise NotImplementedError

    if k_shot is not None:
        if k_shot>0:
            #make each task k-shot
            assert args.regime=='latent_ER', f'Not implemented for {args.regime} yet, should check the task type in select_fewshot_subset_per_class'
            scenario = select_fewshot_subset_per_class(scenario, k_shot) #make each task k-shot
    if args.permute_task_order: 
        assert args.dataset_name!='MNIST_bckgrndwap' 
        if scenario.nb_tasks > 1:
        # random permutation of task order
            scenario = create_subscenario(scenario, args.task_order)
            scenario_test = create_subscenario(scenario_test, args.task_order)
    return scenario, scenario_test
         
def estimate_compute_regime(loader:DataLoader,model:nn.Module, epochs:int, estimate_time=False)->UsedFlops:
    # start = time.time()
    backward_possible=(sum([p.requires_grad for p in model.parameters()])>0 and model.training)
    if 'cuda' in device:
        torch.cuda.synchronize()
    start = time.time()
    if estimate_time:
        for x,y,_ in loader:
        # x,y,_ = next(iter(loader))
            x=x.to(device)
            y=y.to(device)
            pred = model(x)  
            if backward_possible:
                loss=F.cross_entropy(pred,y)
                loss.backward()
                model.optimizer.step()
                model.zero_grad(set_to_none=True)
        if 'cuda' in device:
            torch.cuda.synchronize()
        end = time.time()
            # time_per_batch = end-start
        time_per_epoch= end-start #=time_per_batch*len(loader)
        time_per_task = time_per_epoch*epochs
    else:
        time_per_epoch=0
        time_per_task=0
    x=next(iter(loader))[0]
    x=x.to(device)
    try:         
        flops = FlopCountAnalysis(model, x)
        flops.tracer_warnings('none')
        flops_per_batch_fw = flops.total()
    except Exception as ex:
        flops_per_batch_fw=0
    flops_per_batch_bw = flops_per_batch_fw*2
    #per epoch  
    flops_per_epoch_fw=flops_per_batch_fw*len(loader)
    flops_per_epoch_bw=flops_per_batch_bw*len(loader)

    #per task
    flops_per_task_fw=flops_per_epoch_fw*epochs
    flops_per_task_bw=flops_per_epoch_bw*epochs       

    results_compute=UsedFlops(total_gigaflops_fw=flops_per_task_fw/10e9,
                        total_gigaflops_bw=flops_per_task_bw/10e9, 
                        total_gigaflops=flops_per_task_fw/10e9+flops_per_task_bw/10e9,          
                        total_gigaflops_fw_es = 0, 
                        total_gigaflops_bw_es = 0, 
                        total_gigaflops_es = 0,  
                        walltime_task=time_per_task,
                        total_flops_fw_epoch=flops_per_epoch_fw, 
                        total_flops_bw_epoch=flops_per_epoch_bw,
                        walltime_epoch=time_per_epoch)

    return results_compute
    
class DatasetEncoderContinuum(ModelContainer):                               
    def __init__(self, model_name, args_model: ModelContainer.Options, arg_general: ArgsGenerator, input_size: int = None, name_sufix='') -> None:
        args_model = copy.copy(args_model)
        
        self.learn_bn_stats_before_encode = arg_general.learn_bn_stats_before_encode
        if self.learn_bn_stats_before_encode:
            name_sufix+='_learn_bn_stats_1_'


        if input_size is not None:
            args_model.in_size=int(input_size)
        if arg_general.dataset_encoder_name!='fc':
            args_model.flatten_image=False
        args_model.multihead=False
        # args_model.encoder = arg_general.dataset_encoder
        args_model.encoder_name = arg_general.dataset_encoder_name
        args_model.classifier_type = None
        self.encoding_batch_size = arg_general.encoding_batch_size
        ###########
        #do encoding always on cuda if available
        print("DatasetEncoder", device)
        super().__init__(args_model, device=device) #device)
        if not args_model.encoder_name=='multiple':
            self.init_model()
            log_wandb({'model/n_params_dataset_encoder': sum(p.numel() for p in self.model.feature_extractor.parameters())})
        self.arg_general=arg_general

        if self.transforms is None:    
            self.transforms = arg_general.dataset.dataset_info.transformations
        if self.transforms_val is None:    
            self.transforms_val = arg_general.dataset.dataset_info.transformations_val
        if arg_general.dataset_encoder_name=='fc':
            self.train_data_file =  f"{self.arg_general.dataset_name}_train_{model_name}_pretrained_{self.args.pretrained_encoder}_fc{self.args.feature_size}{name_sufix}.hdf5"
            self.test_data_file =  f"{self.arg_general.dataset_name}_test_{model_name}_pretrained_{self.args.pretrained_encoder}_fc{self.args.feature_size}{name_sufix}.hdf5"
        elif arg_general.dataset_encoder_name in ['ViT-B/32','ViT-B/16']:
            self.train_data_file =  f"{self.arg_general.dataset_name}_train_{model_name}_pretrained_{self.args.pretrained_encoder}_{self.args.vit_out_layer}{name_sufix}.hdf5"
            self.test_data_file =  f"{self.arg_general.dataset_name}_test_{model_name}_pretrained_{self.args.pretrained_encoder}_{self.args.vit_out_layer}{name_sufix}.hdf5"
        else:
            self.train_data_file =  f"{self.arg_general.dataset_name}_train_{model_name}_pretrained_{self.args.pretrained_encoder}{name_sufix}.hdf5"
            self.test_data_file =  f"{self.arg_general.dataset_name}_test_{model_name}_pretrained_{self.args.pretrained_encoder}{name_sufix}.hdf5"
               
        self.train_data_file=self.train_data_file.replace("/","_")
        self.test_data_file=self.test_data_file.replace("/","_")
        
        self.estimate_compute_regime = self.arg_general.estimate_compute_regime_encoding
        self.debug=arg_general.debug
        self.meter_flops=SumMeter()
        self.n_encoding_forward_pathes=0
        self.flops_per_batch=None
        self.model.pbar=None
     
    def _encode_scenario(self, scenario, filename, pref='',estimate_compute:bool=None, encoded=False):           
        """
        Encodes 'scenario' into file stored in the path provided in 'filename'
        if 'encoded' is True assume that data is already encoded and only needs to be stored in a file
        if 'estimate_compute' is True, only estimates compute and does not do the encoding
        """
        with torch.no_grad():
            if estimate_compute is None:
                estimate_compute=self.estimate_compute_regime

            def encode_with_compute_estimation(model,x):
                x=x.to(self.device)                  
                if model.pbar is not None:
                    model.pbar.update(len(x)) 
                if model.flops_per_batch is None:   
                    model.encoding_batch_size=x.shape[0]
                    #estimate flopws per batch
                    flops = FlopCountAnalysis(model, x)
                    flops.tracer_warnings('none')
                    try:
                        model.flops_per_batch=flops.total()
                    except:
                        model.flops_per_batch=0
                if x.shape[0]!=model.encoding_batch_size:
                    model.encoding_batch_size=x.shape[0]
                    #estimate flopws per batch
                    flops = FlopCountAnalysis(model, x)
                    flops.tracer_warnings('none')
                    try:
                        model.flops_per_batch=flops.total() 
                    except:
                        model.flops_per_batch=0   
                self.meter_flops.add(model.flops_per_batch/10e9)
                return model(x)
            
            if not encoded:
                self.pbar = tqdm(range(int(sum([len(n_t) for n_t in scenario]))), desc=f'encoding {self.arg_general.dataset_name} with {self.args.encoder_name}')
                inference_fct = encode_with_compute_estimation #(lambda model, x: model(x.to(device)))
                was_training = self.training
                self.eval()
                print(self.device)
                self.to(self.device) 
                if estimate_compute:
                    #only estimates compute and does not do encoding
                    Gigaflops_stream_encodng=0
                    walltime_encoding=0
                    for _, taskset in enumerate(scenario): 
                        loader = DataLoader(taskset,self.encoding_batch_size)
                        result = estimate_compute_regime(loader,self,1)
                        Gigaflops_stream_encodng+=result.total_gigaflops_fw
                        walltime_encoding+=result.walltime_task
                    log_wandb({f'compute/Gigaflops_stream_encodng_{pref}':Gigaflops_stream_encodng})
                    log_wandb({f'compute/walltime_encoding_{pref}':walltime_encoding})
                    return

                if self.learn_bn_stats_before_encode:
                    # was_training=self.training
                    self.train()
                    #scenario has 1 task in it
                    for _, taskset in enumerate(scenario): 
                        loader = DataLoader(taskset,self.encoding_batch_size)
                        for e in range(5):
                            print('warmup epoch ',e)
                            for x,y,t in loader:
                                x=x.to(device)
                                _ = self(x)    
                self.eval()
                with torch.no_grad():
                    if 'cuda' in device:
                        torch.cuda.synchronize()
                    start = time.time()
                    encoded_scenario = encode_scenario(model=self,
                                                scenario=scenario,
                                                batch_size=self.encoding_batch_size,
                                                filename=filename,
                                                inference_fct=inference_fct)
                    if 'cuda' in device:
                        torch.cuda.synchronize()
                    end = time.time()
                log_wandb({f'compute/Gigaflops_stream_encodng_{pref}':self.meter_flops.value()[0]})
                log_wandb({f'compute/walltime_encoding_{pref}':end-start})
                self.pbar.close()
                if was_training:
                    self.train()
            else:
                #just write data into file 
                encoded_dataset = None
                for task_id, taskset in enumerate(scenario):
                    x, y, t = taskset.get_raw_samples()
                    if encoded_dataset is None:
                        encoded_dataset = H5Dataset(x, y, t, data_path=filename)
                    else:
                        encoded_dataset.add_data(x, y, t)


    def load_dataset(self, *args, **kwargs):    
        return self.load_scenarios(*args, **kwargs)
    
    def _load_scenario(self, scenario, dir, file_name, sufix='', encoded=False): 
        #deal with encoding performed by possible concurent runs, encode the scenario first into a temporary folder.
        n=wandb.util.generate_id()
        dir_temp=f'{dir}/_temp_{n}/'        
        file_destination = f'{dir}/{file_name}'
        file_temp = f"{dir_temp}/{file_name}"
        assert not os.path.exists(dir_temp)
        os.makedirs(f'{dir_temp}')

        self._encode_scenario(scenario, file_temp, sufix, encoded=encoded)
        
        #move and remove
        if not os.path.isfile(file_destination):
            shutil.move(file_temp, dir+f"/{file_name}")
        shutil.rmtree(dir_temp)

        if not self.estimate_compute_regime:
            data = self._load_H5Dataset(file_destination)
        else:  
            data=None
        return data
    
    def _load_H5Dataset(self, file)->H5Dataset:
        file_name=file.split('/')[-1]
        #on mila cluster copy to $SLURM_TMPDIR and load the dataset from $SLURM_TMPDIR
        if 'SLURM_TMPDIR' in os.environ:
            temp_dir = os.environ.get('SLURM_TMPDIR')
            shutil.move(file, temp_dir+f"/{file_name}")
            # os.system(f"cp -r {file} {temp_dir}")
            data = H5Dataset(x=None, y=None, t=None, data_path=f'{temp_dir}/{file_name}')
        else:
            data = H5Dataset(x=None, y=None, t=None, data_path=f"{file}")
        return data

    @staticmethod   
    def _check_h5_encoded_scenario(encoded_scenario:H5Dataset, original_scenario:ClassIncremental):
        if len(encoded_scenario) != len(original_scenario.dataset[0]):
            return False

        # for task_set_enc, task_set in zip(encoded_scenario, original_scenario):
        #     if  len(task_set_enc) != len(task_set):
        #         return False
        return True

    def load_scenarios(self, scenario, scenario_test, data_path, regenerate=False, run=None, encoded=False): 
        """
        Loads or encodes 'scenario' and 'scenario_test' into 'data_path'.
        """        
        nb_tasks=scenario.nb_tasks
        assert nb_tasks==1 #enocding is done with number of tasks 1
        #if encoded is True, assume that data in scenario is already encoded and only need to be stored in a file
        
        #just to make sure all scenarios are encoded with the same number of tasks (should be 1)
        self.train_data_file = self.train_data_file.split('.')[0]+f'_ntasks_{nb_tasks}'+".hdf5"
        self.test_data_file = self.test_data_file.split('.')[0]+f'_ntasks_{nb_tasks}'+".hdf5"
        
        dir = f'{data_path}/EncodedDatasets/'
        train_file=f"{dir}/{self.train_data_file}"
        test_file=f"{dir}/{self.test_data_file}"

        if not os.path.exists(dir):
            os.makedirs(dir)

        if regenerate:
            os.system(f'rm {train_file}')
            os.system(f'rm {test_file}')
        else:
            if run is not None and not self.estimate_compute_regime:
                if os.path.isfile(test_file) and os.path.isfile(train_file):
                    #if both encoding already exist, dont create a new run
                    run.delete()
                    return      
                      
        #trainin data
        if not os.path.isfile(train_file) or self.estimate_compute_regime:               
            data_train = self._load_scenario(scenario=scenario,dir=dir,file_name=self.train_data_file, sufix='train', encoded=encoded)
        else:
            # file exists
            try:
                data_train = self._load_H5Dataset(train_file)
                # try loading the file
                if not self._check_h5_encoded_scenario(data_train,scenario) and not self.debug:
                    print(f"Sscenario {train_file} must be corrupted. I delete the file and reencode")
                    os.remove(train_file)
                    #regenerate
                    data_train = self._load_scenario(scenario,dir,self.train_data_file,'train',encoded)
                else:
                    if not self.debug:
                        #just estimate compute
                        _ = self._encode_scenario(scenario, '',pref='train', estimate_compute=True, encoded=encoded)
            except:
                print("Can not load the file, should be corrupted. I delete the file")
                os.remove(train_file)
                data_train = self._load_scenario(scenario,dir,self.train_data_file,'train',encoded)       
        print(f'loaded from {train_file}') 

        #test data 
        if not os.path.isfile(test_file) or self.estimate_compute_regime:       
            data_test = self._load_scenario(scenario_test,dir,self.test_data_file,'test',encoded)
        else:
            try:
                data_test = self._load_H5Dataset(test_file)
                # data_test = H5Dataset(x=None, y=None, t=None, data_path=test_file)
                if not self._check_h5_encoded_scenario(data_test,scenario_test) and not self.debug:
                    print(f"Sccenario {test_file} must be corrupted. I delete the file and reencode")
                    os.remove(test_file)
                    data_test = self._load_scenario(scenario_test,dir,self.test_data_file,'test',encoded)
                else:
                    if not self.debug:
                        #just estimate compute
                        _ = self._encode_scenario(scenario_test, '',pref='test', estimate_compute=True, encoded=encoded)

            except:
                print("Can not load the file, should be corrupted. I delete the file")
                os.remove(test_file)
                data_test = self._load_scenario(scenario_test,dir,self.test_data_file,'test',encoded)
        
        print(f'loaded from {test_file}')
        return data_train, data_test

if __name__== "__main__":            
    # example arguments
    # "--debug", "1", "--pretrained_encoder", "1","--regime", "latent_ER", "--dataset_name", "CIFAR100", "--n_tasks", "1", "--dataset_encoder_name", "RN50_clip"                      
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGenerator, dest="Global")
    parser.add_arguments(ModelContainer.Options, dest="Model")
    args = parser.parse_args()

    args_generator:ArgsGenerator = args.Global 
    args_model = args.Model

    wandb_project = args_generator.wandb_project if not args_generator.debug else 'test'
    if args_generator.group_name=='':
        args_generator.group_name = wandb.util.generate_id()
    if not is_connected():
            print('no internet connection. Going in dry')
            os.environ['WANDB_MODE'] = 'dryrun'
    run = wandb.init(project=wandb_project, notes=args_generator.wandb_notes, settings=wandb.Settings(start_method="fork"), reinit=False)
    
    wandb.config.update(args_generator, allow_val_change=True)  
    wandb.config.update(args_model, allow_val_change=True)      
    
    set_seed(manualSeed=args_generator.seed)
    in_size=None                                                           
    if args_model.in_size is None and args_generator.dataset_encoder_name=='fc':
        in_size = np.prod(args_generator.dataset_in_size)
    
    ###################################
    # this does all the job of encoding and creating datalaoders 
    scenario, scenario_test = prepare_scenarios(args_generator,args_model) #,transformations=encoder.transforms,transforms_val=encoder.transforms_val)
    for task_id, (train_taskset, test_taskset) in enumerate(zip(scenario, scenario_test)):
        x,y,t = train_taskset[0]
        print(f"task {task_id}, x in {x.shape}, y {y}")             
    ###################################
    # #under the hood it creates encoder and datasets like this  
    # encoder = DatasetEncoderContinuum(args_generator.dataset_encoder_name,args_model, args_generator, input_size=in_size)    
    # scenario, scenario_test = prepare_scenarios_sample(args_generator,args_model,transformations=encoder.transforms,transforms_val=encoder.transforms_val, n_tasks=1, increment=0, dataset_name=args_generator.dataset_name) 
    # dataset_train, dataset_test = encoder.load_scenarios(scenario=scenario, scenario_test=scenario_test, data_path=args_generator.data_path, regenerate=args_generator.regenerate_encodings)
    # ###################################
    # print(len(dataset_train))
    # print(len(dataset_test))  
    # print(dataset_train.data_path)
    # print(dataset_test.data_path)
    ###################################


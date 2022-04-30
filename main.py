
import copy
import os
import time
import warnings
from collections import defaultdict, namedtuple
from enum import unique
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union
from continuum.rehearsal.memory import RehearsalMemory
from continuum.tasks.base import BaseTaskSet

import numpy as np
import torch     
from itertools import cycle
import torch.nn.functional as F
import torch.multiprocessing as mp
from fvcore.nn import FlopCountAnalysis  
from simple_parsing import ArgumentParser, choice
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, dataloader
import wandb
from continuum.tasks.task_set import TaskSet
from continuum.scenarios import _BaseScenario
from args import ArgsGenerator, ExperimentState
from continuum.tasks import TaskType, concat #, split_train_val
# from continuum.tasks.base import BaseTaskSet
from continuum.tasks.h5_task_set import H5TaskSet 
from dataset_encoder import UsedFlops, estimate_compute_regime, prepare_scenarios
from Models import Classifier_options
from torchvision.transforms.transforms import Compose, ToTensor
from Models.model import (ModelContainer, ModelContainer_ER, TorchModuleWrapper)
from Utils.utils import SumMeter, is_connected, log_wandb, set_seed, split_er_buffer, get_scenario_remapping, preprocess_tensor, Logger, test, FlopsMeter,  prepare_dataloader, split_train_val, prepare_balanced_train_loader, prepare_randombuffer_train_loader, convert_to_task_set, TinyDB_hp_tracker
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#used_flops_tuple=namedtuple("ResultTuple",['total_gigaflops_fw','total_gigaflops_bw','total_gigaflops','total_gigaflops_fw_es','total_gigaflops_bw_es', 'total_gigaflops_es','walltime_task','total_flops_fw_epoch', 'total_flops_bw_epoch', 'walltime_epoch'])
loss_function = nn.CrossEntropyLoss()

def init_flops_meters()-> Tuple[FlopsMeter,FlopsMeter]:   
    flopws_meter_pertask_cv = FlopsMeter(['Gigaflops_sofar_fw(including_pt_cv)','Gigaflops_sofar_bw(including_pt_cv)','Gigaflops_sofar(including_pt_cv)',
                                        'Gigaflops_sofar_fw_erlst(including_pt_cv)', 'Gigaflops_sofar_bw_erlst(including_pt_cv)', 'Gigaflops_sofar_erlst(including_pt_cv)', 'walltime_task(including_pt_cv)'])
    flopws_meter_optimal_params = FlopsMeter(['Gigaflops_sofar_fw','Gigaflops_sofar_bw','Gigaflops_sofar',
                                        'Gigaflops_sofar_fw_erlst', 'Gigaflops_sofar_bw_erlst', 'Gigaflops_sofar_erlst', 'walltime_task'])
    return flopws_meter_pertask_cv, flopws_meter_optimal_params

def train_sklearn_classifier(model, task_id, train_loader, test_loaders, flops_meters:FlopsMeter):
    if isinstance(model.classifiers, Iterable):
            model_logreg = model.classifiers[task_id]
    else:
        model_logreg = model.classifiers              
    #collect data
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    for x,y,_ in train_loader: 
        x,y=preprocess_tensor(model,x,y)
        x_train.append(x)
        y_train.append(y)
    for x,y,_ in test_loaders[-1]:
        x,y=preprocess_tensor(model,x,y)
        x_test.append(x)
        y_test.append(y)
    x_train,y_train=torch.cat(x_train).squeeze().cpu().numpy(),torch.cat(y_train).cpu().numpy()
    x_test,y_test=torch.cat(x_test).squeeze().cpu().numpy(),torch.cat(y_test).cpu().numpy()
    #this throws an error somehow
    # x_train, y_train, _ = train_loader.dataset.get_raw_samples()
    # x_test, y_test, _ = test_loaders[-1].get_raw_samples()
    model_logreg.fit(x_train,y_train)
    # Evaluate using the logistic regression classifier
    predictions = model_logreg.predict(x_test)
    accuracy_valid = np.mean((y_test == predictions).astype(float)) #* 100.
    print(f"Accuracy = {accuracy_valid:.3f}")
    # result=used_flops_tuple(*[v for v in flops_meters.value().values()],0,0)
    return model, accuracy_valid, None

def train_model(model_container:ModelContainer_ER,  args:ArgsGenerator, model_name:str, task_id:int, model:TorchModuleWrapper, train_loader:DataLoader, val_loader:DataLoader, train_loaders:List[DataLoader]=None, test_loaders:List=None, lr_anneal=None, epochs=None):
    #task level cross validation
    if lr_anneal is None:
        lr_anneal=model_container.args.lr_anneal
    best_valid_acc=0.
    best_model= None   
    n_epochs_without_improvement = 0
    early_stopped = 0
    flops_meters = FlopsMeter(['flops_task_forward','flops_task_backward', 'flops_task', 'flops_task_forward_es', 'flops_task_backward_es', 'flops_task_es'])
    flops_per_epoch_fw=0
    flops_per_epoch_bw=0
    time_per_batch = None

    if model_container.args.classifier_type in ['logistic_regression', 'random_forrest', 'knn']:
        return train_sklearn_classifier(model, task_id, train_loader, test_loaders, flops_meters)
    else:
        if epochs is None:  
            epochs = args.epochs
        convergence_epoch=epochs
        best_model_epoch=epochs
        flops_per_batch_fw=defaultdict(lambda: None)
        flops_per_batch_bw=defaultdict(lambda:None)
        lr_scheduler=None
        if model.optimizer is not None:     
            for pg_i, param_group in enumerate(model.optimizer.param_groups): 
                param_group["lr"] = model.lr
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=args.epochs)

        if model.optimizer is None:
            epochs = 1        
              
        len_train_loader=len(train_loader) 
        _train_loader=train_loader
        for e in range(epochs):

            if args.er_buffer_type=='balanced_batch' and task_id>0 and args.er_size_per_class>0:
                buffer_train=model_container.replay_buffer
                if isinstance(buffer_train, RehearsalMemory):
                    mem_x, mem_y, mem_t = buffer_train.get()
                elif isinstance(buffer_train, BaseTaskSet): 
                    mem_x, mem_y, mem_t = buffer_train.get_raw_samples()
                if isinstance(mem_x[0], str):
                    args.tasktype=TaskType.IMAGE_PATH

                train_taskset=_train_loader.dataset
                replay_taskset = TaskSet(x=mem_x, y=mem_y, t=mem_t, target_trsf=train_taskset.target_trsf, trsf=train_taskset.trsf, data_type=args.tasktype)
                replay_dataloader = prepare_dataloader(args, replay_taskset, val_split=0, num_workers=2, shuffle=True)[0]
                def ziped_iterator(loader_1, loader_2):
                    for x_1, y_1, t_1 in loader_1:
                        x_2, y_2, t_2 = next(iter(loader_2))
                        x_2=x_2.squeeze()
                        x=torch.cat([x_1, x_2])
                        y=torch.cat([y_1, y_2])
                        t=torch.cat([t_1, t_2])
                        idx = torch.randperm(len(y))
                        x,y,t=x[idx],y[idx],t[idx]
                        yield x,y,t
                if len(replay_dataloader)<len(_train_loader):
                    replay_dataloader=cycle(replay_dataloader)
                train_loader = ziped_iterator(_train_loader, replay_dataloader)
            
            if model.optimizer is not None:  
                for pg_i, param_group in enumerate(model.optimizer.param_groups):
                    log_wandb({f'train_during/lr_t{task_id}_{pg_i}':param_group["lr"]})
            acc = 0
            # if args.ddp:        
            #     train_loader.sampler.set_epoch(e) 

            pbar = tqdm(train_loader) 
            if args.estimate_compute_regime:
                model.train()
                result = estimate_compute_regime(train_loader, model, epochs, estimate_time=args.estimate_time)
                return model, best_valid_acc, result
            
            for b_i, (x,y,_) in enumerate(pbar):
                x,y = preprocess_tensor(model, x, y)    
                x,y = x.to(args.device), y.to(args.device)
                model.train()
                if len(x.shape)<2:
                    #in the Flower dataset getting x of wrong dimention for some reason (shape (1024))
                    x=x.unsqueeze(0)

                if args.record_flops:
                    if flops_per_batch_fw[len(y)] is None: #keep track of compute per batch size used
                        flops = FlopCountAnalysis(model, x)
                        flops.tracer_warnings('none')
                        flops_per_batch_fw[len(y)] = flops.total()
                        flops_per_batch_bw[len(y)] = flops_per_batch_fw[len(y)]*2
                    flops_meters.add({'flops_task_forward':flops_per_batch_fw[len(y)],'flops_task_backward':flops_per_batch_bw[len(y)], 'flops_task':flops_per_batch_fw[len(y)]+flops_per_batch_bw[len(y)]})
                    if not early_stopped:
                        flops_meters.add({'flops_task_forward_es':flops_per_batch_fw[len(y)],'flops_task_backward_es':flops_per_batch_bw[len(y)], 'flops_task_es':flops_per_batch_fw[len(y)]+flops_per_batch_bw[len(y)]})
                
                if model_container.args.multihead:
                    raise NotImplementedError
                    y-=exp_state.n_unique_classes[-1] #to make it multi head compatible
                model.zero_grad(set_to_none=True)
                # try:
                start=time.time()
                logits = model(x, task_id, y=y, epoch=e)
                if args.er_buffer_type=='balanced_batch':
                    count=np.bincount(y.cpu(), minlength=logits.shape[1])+1e-10
                    weights =torch.tensor((max(count)/count)/sum((max(count)/count)), device=args.device).float()
                    loss = F.cross_entropy(logits,y, weight=weights)
                else:
                    loss = loss_function(logits,y)
                if not model.optimizer is None and loss.requires_grad:
                    loss.backward()
                    model.optimizer.step()

                end=time.time()
                if time_per_batch is None:
                    time_per_batch=end-start

                acc_c = torch.sum(logits.max(1)[1] == y).float()/len(y) 
                acc += acc_c
                pbar.set_description("Acc %s" % (acc.cpu().item()/(b_i+1)))
                if args.debug and args.regime=='sample_ER' and b_i ==5:
                    break
            if lr_anneal and lr_scheduler is not None:
                lr_scheduler.step()          
            pbar.close()
            print('model',model_name,'task',task_id, 'train acc: ',acc/len_train_loader, 'epoch: ',e,'\n')
            log_wandb({f'train_during/train_acc_during_t{task_id}':acc/len_train_loader})
            
            if e%args.test_every==0 and test_loaders is not None:
                test_acc_current_task = test(model, task_id, test_loaders[-1], rank=args.device, args=args)  
                print('model',model_name,'task',task_id, 'test acc: ',test_acc_current_task, 'epoch: ',e,'\n')  
                log_wandb({f'test_during/test_acc_during_t{task_id}':test_acc_current_task})
            if e%args.validate_every==0:
                valid_acc = test(model, task_id, val_loader, rank=args.device, args=args)
                print('model',model_name,'task',task_id, 'valid acc: ',valid_acc, 'epoch: ',e,'\n')
                log_wandb({f'valid_during/valid_acc_during_t{task_id}':valid_acc})
                if valid_acc>best_valid_acc: 
                    n_epochs_without_improvement=0
                    best_valid_acc=valid_acc
                    best_model = model.state_dict()
                    best_model_epoch=e
                else:
                    n_epochs_without_improvement+=1  
                    if n_epochs_without_improvement>=args.early_stopping_patience and not early_stopped:
                        early_stopped=1
                        if args.early_stopping:
                            break
                        convergence_epoch=e

        log_wandb({f'convergence_epoch_{task_id}':convergence_epoch})      
        log_wandb({f'best_model_epoch_{task_id}':best_model_epoch})  
        if epochs>1 and best_model is not None:
            # if best_model is None:
            #     best_model = model.state_dict()
            model.load_state_dict(best_model)
        time_per_epoch=0
        time_per_task=0
        if time_per_batch is not None: 
            time_per_epoch=time_per_batch*len_train_loader
            time_per_task=time_per_epoch*epochs
        else:  
            time_per_batch=0

        result_compute = UsedFlops(total_flops_bw_epoch=flops_per_epoch_bw,
                                  total_flops_fw_epoch=flops_per_epoch_fw,
                                  total_gigaflops=flops_meters.meters['flops_task'].value()[0]/10e9, 
                                  total_gigaflops_bw=flops_meters.meters['flops_task_backward'].value()[0]/10e9,
                                  total_gigaflops_fw=flops_meters.meters['flops_task_forward'].value()[0]/10e9,
                                  total_gigaflops_es=flops_meters.meters['flops_task_es'].value()[0]/10e9, 
                                  total_gigaflops_bw_es=flops_meters.meters['flops_task_backward_es'].value()[0]/10e9, 
                                  total_gigaflops_fw_es=flops_meters.meters['flops_task_forward_es'].value()[0]/10e9, 
                                  walltime_epoch=time_per_epoch,
                                  walltime_task=time_per_task)            
        return model, best_valid_acc, result_compute

def learn_task(model_container:ModelContainer_ER, args:ArgsGenerator, model_name:str, task_id:int, train_loader:DataLoader, val_loader:DataLoader,  train_loaders:List[DataLoader]=None, test_loaders:List[DataLoader]=None, params_table:wandb.Table=None, world_size=None):
    # model=model_container.model
    best_valid_acc=0
    best_model = None  
    best_result_flops = None
    best_args_model, best_args_classifier = None, None 
    #iterate over agruments to validate for per task cross validation
    for cv_run_idx, (args_model, args_classifier, model) in enumerate(model_container.get_model_for_training(task_id)):
        print(f'testing with arguments {args_model} and classifier arguments {args_classifier}')
        if args_model is not None:
            lr_anneal=args_model.lr_anneal
        else:
            lr_anneal = model_container.args.lr_anneal
        #task evel cross validation
        if args.schedule == 'classifier+normal':
            #first trin the classifier and then the feature encoder
            assert args.regime != 'latent_ER'
            #fix feature extractor and finetuen only classifier
            model.freeze_feature_extractor()
            model, valid_acc,result_flops = train_model(model_container, args, model_name, task_id, model, train_loader, val_loader, train_loaders, test_loaders, lr_anneal=lr_anneal)
            #finetune both     
            model.freeze_feature_extractor(False)
            model, valid_acc,result_flops = train_model(model_container, args, model_name, task_id, model, train_loader, val_loader, train_loaders, test_loaders, lr_anneal=lr_anneal)
        else:
            if args.n_epochs_task_level_cv is not None and args.task_level_cv and not (args.keep_best_params_after_first_task and task_id>0) :
                epochs=args.n_epochs_task_level_cv
            else:
                epochs=None
            if args.finetuning_only_norms:
                model.freeze_feature_extractor()
                model.unfreeze_bn()  
                if model_container.args.encoder_name=='ViT-B/16':
                    if args.unfreeze_input_layer:
                        model.unfreeze_first()    
            if model_container.args.encoder_name=='ViT-B/16' and args.freeze_vit_untill_layer is not None:
                model.freeze_vit_untill_layer(args.freeze_vit_untill_layer)
            model, valid_acc, result_flops = train_model(model_container, args, model_name, task_id, model, train_loader, val_loader, train_loaders, test_loaders, epochs=epochs, lr_anneal=lr_anneal)

        flopws_meter_pertask_cv.add([result_flops.total_gigaflops_fw, result_flops.total_gigaflops_bw, result_flops.total_gigaflops, result_flops.total_gigaflops_fw_es,
                                     result_flops.total_gigaflops_bw_es, result_flops.total_gigaflops_es, result_flops.walltime_task])#[:7])
        flopws_meter_pertask_cv.log_flops(task_id)

        if valid_acc>=best_valid_acc:
            best_result_flops=result_flops
            best_model = model.create_checkpoint()        
            best_args_model, best_args_classifier = args_model, args_classifier
            best_valid_acc=valid_acc

    if model_container.args.classifier_type in ['slda', 'nmc']:
        return model

    if best_args_model is None:        
        best_result_flops=result_flops 
        best_model = model.create_checkpoint()      
        best_args_model, best_args_classifier = args_model, args_classifier
        best_valid_acc=valid_acc

    model_container.reset_args(best_args_model, best_args_classifier)

    if args.n_epochs_task_level_cv is not None and args.task_level_cv and not (args.keep_best_params_after_first_task and task_id>0):
        #train for total epochs with best hps
        model, _ = model_container.create_model(best_args_model, best_args_classifier)
        model.load_state_dict(model_container.model.state_dict())
        if args.regime=="latent_ER" and model_container.args.classifier_type=='clip_0_shot':
            model.feature_extractor=nn.Identity()
        model.set_optimizer()
        model, best_valid_acc, result_flops = train_model(model_container, args, model_name, task_id, model, train_loader, val_loader, train_loaders, test_loaders)
        best_result_flops=result_flops  
        best_model = model.create_checkpoint()
        flopws_meter_pertask_cv.add([best_result_flops.total_gigaflops_fw, best_result_flops.total_gigaflops_bw, best_result_flops.total_gigaflops, best_result_flops.total_gigaflops_fw_es,
                                     best_result_flops.total_gigaflops_bw_es, best_result_flops.total_gigaflops_es, best_result_flops.walltime_task])
        flopws_meter_pertask_cv.log_flops(task_id)

    # if best_args_model is not None: 
    print(f"selected arguments model task {task_id}", best_args_model)      
    # if best_args_classifier is not None:
    print(f"selected arguments classifier task {task_id}", best_args_classifier) 
    # if params_table is not None:
    params_table.add_data(str(task_id), str(best_args_model), str(best_args_classifier))   

    #save best parameters to hp database
    if args.use_hp_database and args.task_level_cv:  
        if task_id==0 and args.keep_best_params_after_first_task:
            hp_db.log_into_hp_ds(args_global,best_args_model,best_args_classifier)

    if best_result_flops is not None:     
        ###log flops (wandb does the accumulation)###      
        for n,v in best_result_flops.__dict__.items():
            log_wandb({f'compute/{n}':v})
        #############################################    
        flopws_meter_optimal_params.add([best_result_flops.total_gigaflops_fw, best_result_flops.total_gigaflops_bw, best_result_flops.total_gigaflops, best_result_flops.total_gigaflops_fw_es,
                                        best_result_flops.total_gigaflops_bw_es, best_result_flops.total_gigaflops_es, best_result_flops.walltime_task])
        flopws_meter_optimal_params.log_flops(task_id)
        #log per epoch walltime and per epoch compute
        log_wandb({f'compute/walltime_task_t{task_id}':best_result_flops.walltime_task})
        log_wandb({f'compute/walltime_per_epoch_t{task_id}':best_result_flops.walltime_epoch})
        log_wandb({f'compute/flops_per_epoch_t{task_id}': best_result_flops.total_flops_fw_epoch+best_result_flops.total_flops_bw_epoch})

    model_container.init_model()
    if best_model is None:   
        best_model = model.create_checkpoint()  
    model_container.model.train()
    model_container.model.load_checkpoint(best_model)
    model_container.model.set_optimizer()

    if model_container.args.classifier_type in ['slda', 'nmc']:
        #mnake sure slda and nmc properly store their state dicts  
        accuracy=test(model_container, task_id, val_loader, args=args, rank=args.device)
        if not best_valid_acc==accuracy:
            #temporal strange fix for the problem
            model_container.model.load_checkpoint(best_model)
            accuracy=test(model_container, task_id, val_loader, args=args, rank=args.device)
        assert best_valid_acc==accuracy 
        # problem with SLDA after second task or so:
        # best_model = model.create_checkpoint()
        # model_container.model.load_checkpoint(best_model)
        # assert test(model, task_id, val_loader, args=args, rank=args.device)==test(model_container, task_id, val_loader, args=args, rank=args.device)
    return model_container.model

def prepare_model_and_scenario(args:ArgsGenerator,args_model:ModelContainer_ER.Options,args_classifier)->Tuple[ModelContainer,_BaseScenario,_BaseScenario]:
    '''
    Prepares the model and the scenarios to train on.
    '''
    if args.regime=='latent_ER':    
        scenario, scenario_test = prepare_scenarios(args, args_model)
        n_classes=[scenario[i].nb_classes for i in range(len(scenario))]  
        model_container=ModelContainer_ER(args_model, args_global=args, args_classifier=args_classifier, n_classes=n_classes, device=args.device)    
        model_container.transforms=None
        model_container.transforms_val=None
        if not args.encode_with_continuum:   
            args_model.in_size = int(scenario.dataset[0].shape[-1])
        model_container.init_model()
    elif args.regime=='sample_ER':
        args_model.in_size = int(args.dataset.dataset_info.size[-1])
        model_container=ModelContainer_ER(args_model, args_global=args, args_classifier=args_classifier, n_classes=args.n_classes, device=args.device)  
        if args_model.flatten_image:
            args_model.in_size = int(np.prod(args.dataset.dataset_info.size[-1]))
        model_container.init_model()
        scenario, scenario_test = prepare_scenarios(args, args_model, transformations=model_container.transforms, transforms_val=model_container.transforms_val)
        if args.permute_task_order:
            n_classes=[scenario[i].nb_classes for i in range(len(scenario))]
            model_container=ModelContainer_ER(args_model, args_global=args, args_classifier=args_classifier, n_classes=n_classes, device=args.device)
            model_container.init_model()
        if args.finetuning_only_norms:
            model_container.model.freeze_feature_extractor()
            model_container.model.unfreeze_bn()
            if model_container.args.encoder_name=='ViT-B/16':
                if args.unfreeze_input_layer:  
                    model_container.model.unfreeze_first()
        if model_container.args.encoder_name=='ViT-B/16':
            if args.freeze_vit_untill_layer is not None: 
                model_container.model.freeze_vit_untill_layer(args.freeze_vit_untill_layer)

        
    return model_container, scenario, scenario_test
    
def create_per_class_prototypes(x,y,t, n_prototypes_per_class, merge_type='mean', n_samples_per_prototype=None):
    """
        Calculate prototypes for prototype based replay
    """
    n=n_prototypes_per_class
    if merge_type == None:
        return x,y,t
    def split_for_prototype(idxs, n, n_samples_per_prototype):  
        random_state = np.random.RandomState(1)
        random_state.shuffle(idxs)
        if n_samples_per_prototype is None:
            #create possibly equal group sizes
            groups_size = [len(idxs) // n + (1 if x < len(idxs) % n else 0)  for x in range (n)]
            #select indicies for group creation without replacement
            groups = [ idxs[sum(groups_size[:max(0,c)]):sum(groups_size[:c])+groups_size[c]] for c in range(len(groups_size)) ]
        else:
            #randomly select indicies fro groups with replacement
            groups = [random_state.choice(idxs,n_samples_per_prototype, replace=True)]
        return groups
    x_, y_, t_ = [],[],[]
    groups_idxs_per_class = [split_for_prototype(np.where(y==c)[0],n,n_samples_per_prototype) for c in np.unique(y)]
    for c_idxs in  groups_idxs_per_class:
        prototypes, label, task = [np.mean(x[i],axis=0) for i in c_idxs], np.concatenate([y[i] for i in c_idxs]), np.concatenate([t[i] for i in c_idxs])
        assert len(np.unique(label))==1
        assert len(np.unique(task))==1
        x_.append(prototypes)
        y_.append([np.unique(label)[0]]*len(prototypes))
        t_.append([np.unique(task)[0]]*len(prototypes))

    return np.concatenate(x_), np.concatenate(y_), np.concatenate(t_)
     
def main(args:ArgsGenerator, args_model:ModelContainer_ER.Options, args_classifier, **kwargs):
    exp_state=ExperimentState()
    test_loaders_sofar=[]
    train_loaders_sofar=[]
    valid_loaders_sofar=[]
    test_accuracies_past=[]
    valid_accuracies_past=[]    
    print(args.device)
    model_container, scenario, scenario_test = prepare_model_and_scenario(args,args_model,args_classifier)  
    print(model_container.model)
    logger=Logger(args,model_container,scenario.nb_tasks)    
    #wandb table for logging best selected hyperparameters per task
    best_cvparams_table = logger.best_cvparams_table

    #######
    # remapping is necessary for permuted order scenarios
    class_mapping = get_scenario_remapping(scenario)
    model_container.set_mapping(class_mapping)        
    for task_id, (train_taskset, test_taskset) in enumerate(zip(scenario, scenario_test)):  
        #######
        if args.dataset_name=='MNIST_bckgrndwap':
            #currently the train_taskset only contains transfroms of the current task, the other ones are None
            #current task_set should contain transfroms for old tasks as well when all tasks are being replayed (maybe this should be fixed on the continuum side)
            train_taskset.trsf=[Compose([*scenario.inc_trsf[t],ToTensor()]) for t in range(len(scenario))]
        ########
        exp_state.current_task=task_id
        log_wandb({'task': task_id})            
                                        
        if isinstance(train_taskset, H5TaskSet):
            #using H5TaskSet was taking to long, probably due to reading from hd5 file, 
            # also maybe the issue is the netowrk connection to scratch on mila cluster, maybe putting in into $TEMP(local to the node) should help
            train_taskset = convert_to_task_set(args,train_taskset)
            train_taskset, val_taskset, train_idxs = split_train_val(train_taskset, val_split=args.valid_fraction, valid_k=args.valid_k)              
            if model_container.transforms_val is not None:
                if isinstance(model_container.transforms_val, List): 
                    transforms_val = Compose(model_container.transforms_val)
                else:
                    transforms_val=model_container.transforms_val
                val_taskset.trsf = transforms_val

        else: 
            train_taskset, val_taskset, train_idxs = split_train_val(train_taskset, val_split=args.valid_fraction, valid_k=args.valid_k) 
            #set validation transfroms instead of training transfroms for val_taskset
            if model_container.transforms_val is not None:
                if isinstance(model_container.transforms_val, List):
                    if not isinstance(model_container.transforms_val[0], List):
                        transforms_val = Compose(model_container.transforms_val)
                    else:
                        transforms_val = Compose(model_container.transforms_val[task_id])
                else:
                    transforms_val=model_container.transforms_val
                val_taskset.trsf = transforms_val
        #######################################
        # Task similarity logging               
        # if args.log_task_similarity and args.regime=='latent_ER':
        #     logger.log_similarity(train_taskset, task_id)
        #######################################
        #for per task hp cv we might want to add some samples from the replay buffer to the validation set
        if args.fraction_buffer_samples_valid>0 and args.er_size_per_class>0 and task_id>0: 
            buffer_train, buffer_valid = split_er_buffer(model_container.replay_buffer, args.fraction_buffer_samples_valid, data_type=val_taskset.data_type)#TaskType.IMAGE_PATH if args.regime=='sample_ER' else TaskType.TENSOR)
            #merge buffer_valid into the validation set
            #will take the transfromations fromt the val_taskset
            val_taskset_for_hp_tuning = concat([val_taskset,buffer_valid])
        else:
            buffer_train=model_container.replay_buffer
            val_taskset_for_hp_tuning=val_taskset
                        
        ########################################
        #handle random replay buffer
        if args.er_size_per_class>0 and task_id > 0:  
            if args.er_buffer_type=='random':                    
                train_loader=prepare_randombuffer_train_loader(args, train_taskset=train_taskset, buffer_train=buffer_train)
            elif args.er_buffer_type=='balanced':
                train_loader=prepare_balanced_train_loader(args, buffer_train, current_task_set=train_taskset)
            # elif args.er_buffer_type=='balanced_batch':
            #     train_loader=prepare_balanced_batch_train_loader(args, buffer_train, train_taskset)
            else:
                train_loader,_=prepare_dataloader(args, train_taskset, val_split=0., num_workers=2, shuffle=True) 
        else:
            train_loader,_=prepare_dataloader(args, train_taskset, val_split=0., num_workers=2, shuffle=True) 
        ########################################

        val_loader,_=prepare_dataloader(args, val_taskset, num_workers=2, val_split=0.)         
        val_loader_hp_tuning,_=prepare_dataloader(args, val_taskset_for_hp_tuning, num_workers=2, val_split=0.)
        test_loader,_=prepare_dataloader(args, test_taskset, num_workers=2, val_split=0.)
        test_loaders_sofar.append(test_loader)
        train_loaders_sofar.append(train_loader)
        valid_loaders_sofar.append(val_loader)
        model_container.ready_for_new_task(task_id=task_id, new_classes=scenario[task_id].get_classes(), expand_single_head=args.dataset_name!='MNIST_bckgrndwap')
        if args.debug:
            print(model_container)
        if args.concat_validation_sets:    
            val_taskset_for_hp_tuning = concat([loader.dataset for loader in valid_loaders_sofar])
            val_loader_hp_tuning,_=prepare_dataloader(args, val_taskset_for_hp_tuning, num_workers=2, val_split=0.)
        if args.concat_test_sets:  
            test_taskset = concat([loader.dataset for loader in test_loaders_sofar])    
            test_loader,_=prepare_dataloader(args, val_taskset_for_hp_tuning, num_workers=2, val_split=0.)
        model_container.model = learn_task(model_container, args, model_container.args.encoder_name, task_id, train_loader, val_loader_hp_tuning, train_loaders_sofar, test_loaders_sofar, params_table=best_cvparams_table)
        test_acc =  test(model_container, task_id, test_loader, args=args, rank=args.device)  
        valid_acc =  test(model_container, task_id, val_loader, args=args, rank=args.device)
        valid_acc_hp_tuning_set =  test(model_container, task_id, val_loader_hp_tuning, args=args, rank=args.device)
        log_wandb({"test_acc_online":test_acc}, step=('task', task_id))#, prefix='test/')
        log_wandb({"valid_acc_online":valid_acc}, step=('task', task_id))#, prefix='test/')
        log_wandb({"valid_acc_online(hp_tuning)":valid_acc_hp_tuning_set}, step=('task', task_id))#, prefix='test/')
        log_wandb({f"test_acc_best_t{task_id}":test_acc}, step=('task', task_id))#, prefix='test/')
        log_wandb({f"valid_acc_best_t{task_id}":valid_acc}, step=('task', task_id))#, prefix='test/')
        test_accuracies_past.append(test_acc)
        valid_accuracies_past.append(valid_acc)
        #add to replay buffer
        if args.er_size_per_class>0:
            x,y,t = scenario[task_id].get_raw_samples()
            if len(y.shape)>1:
                y=y[0]
                t=t[0]
            #only add samples from train set (train_idxs), not the validation set to the replay buffer
            x,y,t = x[train_idxs],y[train_idxs],t[train_idxs]
            print(len(train_idxs))
            if args.er_with_prototypes:
                #create er_size_per_class prototypes
                if args.size_of_random_prototype_factor is not None:
                    size_of_random_prototype = int(np.mean(np.bincount(scenario[task_id]._y))/args.er_size_per_class)
                    size_of_random_prototype*=args.size_of_random_prototype_factor
                    size_of_random_prototype=int(size_of_random_prototype)
                else:
                    size_of_random_prototype=None
                x,y,t = create_per_class_prototypes(x,y,t, n_prototypes_per_class=args.er_size_per_class, n_samples_per_prototype=size_of_random_prototype)
            model_container.add_to_buffer((x,y,t))                 
        logger.log_results(args,task_id,model_container,test_loaders_sofar,valid_loaders_sofar,test_accuracies_past,valid_accuracies_past,best_cvparams_table)
        if args.stop_after_first_task and task_id==0:
            break
        if args.reinit_between_tasks:
            model_container.reinit_model()
        if args_global.save_final_model:
            ckpt_name=f'{model_container.args.encoder_name.replace("/","_")}_trained_{args_global.dataset_name}_task{task_id}_{args_global.n_tasks}tasks_er_size_{args_global.er_size_per_class}'
            if args_model.checkpoint is not None:
                ckpt_name+=args_model.checkpoint
            torch.save({
                'model_state_dict': model_container.model.state_dict(),     
            }, args_global.weights_path+f'/{ckpt_name}.ckpt')
        
        
    flopws_meter_pertask_cv.log_flops()
    flopws_meter_optimal_params.log_flops()
    logger.close()

if __name__== "__main__":  
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGenerator, dest="Global")
    parser.add_arguments(ModelContainer.Options, dest="Model")
    parser.add_arguments(Classifier_options, dest="Classifier")
    args = parser.parse_args()
    args_global:ArgsGenerator = args.Global 
    args_model = args.Model
    if args_model.classifier_type=='fc': 
        args_classifier = args.Classifier.CLS_NN
    elif args_model.classifier_type=='logistic_regression':
        args_classifier = args.Classifier.CLS_LogReg
    elif args_model.classifier_type=='BiT_classifier':
        args_classifier = args.Classifier.CLS_BiT
    elif args_model.classifier_type=='random_forrest':
        args_classifier = args.Classifier.CLS_RF
    elif args_model.classifier_type=='clip_0_shot':
        args_classifier = args.Classifier.CLP_0
    elif args_model.classifier_type == 'knn':
        args_classifier = args.Classifier.CLS_KNN
    elif args_model.classifier_type == 'weightnorm':
        args_classifier = args.Classifier.CLS_WeightNorm
    elif args_model.classifier_type == 'slda':
        args_classifier = args.Classifier.CLS_SLDA
    elif args_model.classifier_type == 'nmc':
        args_classifier = args.Classifier.CLS_NMC
        args_global.device = 'cpu'
    else:
        args_classifier=None  
    
    wandb_project = args_global.wandb_project if not args_global.debug else 'test'
    if args_global.group_name=='':
        args_global.group_name = wandb.util.generate_id()
    # assert is_connected()#:
    if not is_connected():
        print('no internet connection. Going in dry')
        # os.environ['WANDB_MODE'] = 'dryrun'
        wandb_mode="offline"
    else:
        wandb_mode=None
    
    def start_experiment(run):                             
        wandb.config.update(args_global, allow_val_change=True)  
        wandb.config.update(args_model, allow_val_change=True)
        if args_classifier is not None:
            wandb.config.update(args_classifier, allow_val_change=True)                   
        main(args_global, args_model, args_classifier, world_size=torch.cuda.device_count())
        run.finish()

    key=f'{args_global.md5}_{args_model.md5}_{args_classifier.md5}'
    #databse for hyperparameter saving: hp dataset is job specific
    if args_global.use_hp_database:
        if 'SLURM_TMPDIR' in os.environ:     
            args_global.hp_db_path = os.environ.get('SLURM_TMPDIR') 
        hp_db = TinyDB_hp_tracker(args_global.hp_db_path+f'/hp_database_{args_global.dataset_name}.json', key=key)    
    args_global.group_name = args_global.generate_group_name(args_model.md5+args_classifier.md5)        
    set_seed(manualSeed=args_global.seed)  
    if args_global.n_task_order_permutations>0:
        args_global.permute_task_order=1               
        for run_i in range(args_global.n_task_order_permutations):
            flopws_meter_pertask_cv, flopws_meter_optimal_params = init_flops_meters()
            args_global.generate_task_order(run_i)
            if args_global.use_hp_database:
                success,args_global,args_model,args_classifier=hp_db.load_hps_from_db(args_global,args_model,args_classifier)
            run = wandb.init(project=wandb_project, mode=wandb_mode, notes=args_global.wandb_notes, group=args_global.group_name, settings=wandb.Settings(start_method="fork"), reinit=False, entity=args_global.wandb_entity)
            wandb.save('datasets.py')
            start_experiment(run)
    else:
        flopws_meter_pertask_cv, flopws_meter_optimal_params = init_flops_meters()
        if args_global.use_hp_database:
            success,args_global,args_model,args_classifier=hp_db.load_hps_from_db(args_global,args_model,args_classifier)
        run = wandb.init(project=wandb_project, mode=wandb_mode, notes=args_global.wandb_notes, settings=wandb.Settings(start_method="fork"), reinit=False, entity=args_global.wandb_entity)
        wandb.save('datasets.py')
        start_experiment(run)
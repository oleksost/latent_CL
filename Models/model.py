from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, OrderedDict, Tuple, Union
from torch import nn
import numpy as np
import torch
import os
import copy
import clip
import timm    
import hashlib   
# import vissl
import continuum 
from continuum import rehearsal
from fvcore.nn import parameter_count_table
from simple_parsing import choice
from torchvision.transforms.transforms import Lambda
from .helper import torch_NN, bn_eval, getKeysByValue
from .module_wrapper import TorchModuleWrapper
from .encoders import encoders, EncoderTuple
from args import ArgsGenerator
from dataclasses_json import dataclass_json
from .classifiers import ClassifierOptions, sklearn_classifiers,Classifier_nn, Classifier_BiT, CLIPZeroShotClassifier, LogisticRegression_Classifier
from .classifiers import KNeighbors_Classifier, RandomForest_Classifier, NMC_Classifier, NMC_Classifier, WeightNorm_Classifier, SLDA_Classifier

project_home = f"{os.environ.get('LARGE_CL_HOME')}" if "LARGE_CL_HOME" in os.environ else "."        
_models_path = f"{os.environ.get('SCRATCH')}/Weights/" if "SCRATCH" in os.environ else f"{project_home}/Weights/"

class ModelContainer(nn.Module): 
    '''
    This class contructs trainable model containing a feature extractor and a classifier.
    '''   
    @dataclass_json
    @dataclass
    class Options(ClassifierOptions):
        fix_batchnorms_encoder: bool = 0 #-
        freeze_encoder: bool = 0 #-
        hidden_layer_n: Optional[int]=None # for ViT models would extract the representations of the hidden layer instead of the final output leayer (currently only works with ViT models)
        in_size: Optional[int] = None #-
        # clip: bool = 0 #-
        vit_out_layer:str = choice('last_hidden_state', 'last_layer', default='last_layer') #- # only metters if clip=0
        pretrained_encoder: bool = 1 #-
        checkpoint: Optional[str] = None # If checkoint path is given, loeads weights into the encoder from the given checkpoint
        flatten_image: bool = 0 # if 'True' and used together with sample_ER, image is flattened before feeding to the network
        flatten_features: bool = 1 #-
        classifier_type:Optional[str] = choice('None','conv_block', 'fc', 'conv', 'resnet_block', 'alexnet', 'random_forrest', 'logistic_regression', 'clip_0_shot', 'BiT_classifier', 'knn', 'nmc', 'weightnorm', 'slda', default='fc')#-
        encoder_name:str = choice(encoders.keys(), default='None')# used for the model
        feature_size:int=512 # latent size, only used for fc feature encoder for now
        #Optimizer arguments
        lr: float = 0.001 #-
        lr_anneal: bool = False #-
        optimizer: str = choice('adam', 'sgd', default='adam') #-

        lr_classifier: Optional[float] = None #-
        optimizer_classifier: Optional[str] = choice('adam', 'sgd', default=None) #-
        momentum:float = 0.9 #-
        weight_decay:float = 0. #-

        multihead: bool = False #-             
        lrs_cv:List = field(default_factory=lambda: []) #-
        lr_anneals_cv:List = field(default_factory=lambda:[]) #- [True,False]) #-
        weight_decays_cv:List =field(default_factory=lambda:[]) #, 1e-3] #-

        def serializable_copy(self):
            self_copy = copy.copy(self)
            #remote partials
            self_copy.encoder=None
            return self_copy
        @property
        def md5(self):
            self_copy = copy.copy(self)
            return hashlib.md5(str(self_copy).encode('utf-8')).hexdigest()

        def __post_init__(self):
            self.encoder = encoders[self.encoder_name] #getKeysByValue(encoders,self.encoder_name)[0]
            # if self.freeze_encoder:
            #     self.fix_batchnorms_encoder=1
                
        def generate_cv_args(self):            
            #generate parameters for task level cross validation
            lrs_cv=self.lrs_cv  
            weight_decays_cv=self.weight_decays_cv
            if self.encoder is None:
                if len(lrs_cv)==0:
                    lrs_cv=[0.1*self.lr,self.lr, 10*self.lr]
                if len(weight_decays_cv)==0:
                        weight_decays_cv=[0, 1e-4, 1e-2]
                for lr in lrs:
                    for wd in weight_decays_cv: 
                        for anneal in self.lr_anneals_cv:
                            anneal=bool(anneal)
                            wd=float(wd)
                            lr=float(lr)
                            args=copy.copy(self)
                            args.lr=lr
                            args.lr_anneal=anneal
                            args.weight_decay=wd
                            yield args
            else:
                #finetuning   
                if self.encoder_name in ["ViT-B/16", "ViT-B/32"]:
                    if len(lrs_cv)==0:
                        lrs_cv=[0.1*self.lr, self.lr, 10*self.lr] #[2e-05, 2e-04]
                    if len(weight_decays_cv)==0:
                        weight_decays_cv=[0, 1e-4, 1e-2]              
                else:
                    if len(lrs_cv)==0:
                        lrs = [0.1*self.lr,self.lr, 10*self.lr]
                    if len(weight_decays_cv)==0:
                        weight_decays_cv = [0, 1e-4, 1e-3]

                for lr in lrs_cv:
                    for wd in weight_decays_cv: 
                        for anneal in self.lr_anneals_cv:
                            anneal=bool(anneal)
                            wd=float(wd)
                            lr=float(lr)
                            args=copy.copy(self)
                            args.lr=lr
                            args.weight_decay=wd
                            args.lr_anneal=anneal
                            yield args

    def __init__(self, args:Options, args_classifier=None, n_classes:Union[int,List[int]]=None, debug: bool = False, classes_text_description:List[str]=None, device='cuda', ddp=False, default_transfroms=None, default_transfroms_val=None, weights_path=_models_path) -> None:
        super().__init__()
        self.args=ModelContainer.Options(**args.to_dict()) #copy.copy(args)
        self.ddp = ddp
        self.device=device         
        weights_path=weights_path.replace('//','/')
        self.weights_path=weights_path
        self.args_classifier:ClassifierOptions=copy.copy(args_classifier)
        self.classes_text_description = classes_text_description #for clip classifier
        self.debug = debug
        self.classifier_type=self.args.classifier_type
        self.transforms = default_transfroms 
        self.transforms_val = default_transfroms_val
        self.feature_size = self.args.feature_size
        self.feature_extractor_vit = None
        self.model:TorchModuleWrapper = None

        #buffers
        self.register_buffer('_n_tasks',torch.tensor(1))
        if n_classes is not None:
            self.register_buffer('_n_classes',torch.tensor(n_classes))
        else:
            self._n_classes = None
    
    @property
    def n_tasks(self):
        return self._n_tasks.item() 
    @n_tasks.setter
    def n_tasks(self, value:int):
        self._n_tasks = torch.tensor(value)

    @property
    def n_classes(self):
        if self._n_classes is None: return None
        return self._n_classes.item() 
    @n_classes.setter
    def n_classes(self, value:int):
        self._n_classes = torch.tensor(value)

    def CV_args(self):
        '''
        This functions generatescross validation parameters for the model container (can be used for per task CV).
        '''
        if self.model.optimizer is not None:
            for args in self.args.generate_cv_args():             
                for  args_classifier in self.args_classifier.generate_cv_args():
                    yield args, args_classifier
                    
        else:
            for args_classifier in self.args_classifier.generate_cv_args():
                    yield self.args, args_classifier

    
    def forward(self, x, task_id=None, *args, **kwargs):
        if self.args.multihead:
            assert task_id is not None
        else:
            task_id = None
        return self.model( x, task_id, *args, **kwargs)

    def init_model(self): 
        model, feature_size = self.create_model()
        self.feature_size=feature_size
        self.model=model
        self.model.set_optimizer()
        return

    def create_model(self, args:Options=None, args_classifier:ClassifierOptions=None)->Tuple[TorchModuleWrapper,int]:
        '''
        Creates model given arguments.
        '''
        if args is None:
            args=self.args
        if args_classifier is None:
            args_classifier=self.args_classifier

        encoder, feature_size = self.prepare_encoder(args.encoder, self.feature_size)
        
        if encoder is not None:
            if args.freeze_encoder: 
                for p in encoder.parameters():
                    p.requires_grad = False
            if args.fix_batchnorms_encoder:
                encoder.keep_bn_in_eval_after_freeze=True
                bn_eval(encoder)    
            # self.encoder = encoder
            classifier = self.prepare_classifier(self.classifier_type, feature_size,  n_classes=self.n_classes, encoder=encoder, args_classifier=args_classifier)#.to(self.device)
            model = TorchModuleWrapper(encoder, classifier, keep_bn_in_eval_after_freeze=args.fix_batchnorms_encoder, 
                                flatten_image=args.flatten_image,flatten_features=args.flatten_features, optimizer_name=args.optimizer,
                                lr=args.lr, optimizer_name_classifier=args.optimizer_classifier, lr_classifier=args.lr_classifier, momentum=args.momentum, weight_decay=args.weight_decay).to(self.device)
            if self.args.checkpoint is not None:  
                strict=isinstance(model.classifiers, nn.Identity)
                state_dict=torch.load(self.args.checkpoint)['model_state_dict']          
                keys_to_remove = [k for k in state_dict.keys() if 'classifiers.classifier' in k]
                for k in keys_to_remove:
                    state_dict.pop(k)
                model.load_state_dict(state_dict, strict=strict)
            return model, feature_size
        else:
            #no encoder
            # self.encoder = encoder
            classifier =self.prepare_classifier(self.classifier_type, args.in_size, n_classes=self.n_classes, args_classifier=args_classifier)
            if args.optimizer_classifier is not None:
                args.optimizer = args.optimizer_classifier
            if args.lr_classifier is not None:
                args.lr = args.lr_classifier
            if args.classifier_type in sklearn_classifiers:
                model = TorchModuleWrapper(feature_extractor=None, classifiers=classifier,flatten_features=args.flatten_features)
            else:
                model = TorchModuleWrapper(feature_extractor=None, classifiers=classifier,flatten_features=args.flatten_features, 
                                        optimizer_name=args.optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay).to(self.device)
            return model, args.in_size
          
    def prepare_encoder(self, encoder_tupel:EncoderTuple, feature_size:str=512) -> TorchModuleWrapper:
        encoder_name = getKeysByValue(encoders,encoder_tupel)[0]
        if encoder_name=='None':
            encoder=None
        else:   
            partial_encoder = encoder_tupel.partial_encoder
            if encoder_name=='fc':   
                assert self.args.in_size is not None
                assert not self.args.pretrained_encoder       
                model_tuple = partial_encoder(self.args.in_size, feature_size, device=self.device)
            else:
                if self.args.pretrained_encoder not in encoder_tupel.pretrained_options:
                    print(f"Cant used pretrained option {self.args.pretrained_encoder} for {self.args.encoder_name}, using {not self.args.pretrained_encoder} instead")
                
                
                model_tuple = partial_encoder(pretrained=self.args.pretrained_encoder if self.args.pretrained_encoder in encoder_tupel.pretrained_options else not self.args.pretrained_encoder, 
                                                                        transforms=self.transforms, 
                                                                        fix_batchnorms_encoder=self.args.fix_batchnorms_encoder, 
                                                                        models_path=self.weights_path,
                                                                        input_shape=self.args.in_size,
                                                                        hidden_layer=self.args.hidden_layer_n,
                                                                        device=self.device)
            encoder = model_tuple.encoder
            feature_size = model_tuple.latent_dim
            _transforms = model_tuple.transformation
            _transforms_val = model_tuple.transformation_val
            if _transforms is not None:
                self.transforms = _transforms
            if _transforms_val is not None:
                self.transforms_val = _transforms_val
        return encoder, feature_size
    
    def ready_for_new_task(self, task_id, n_new_classes:int, expand_single_head=True):
        self.n_tasks= self.n_tasks + 1
        if expand_single_head:           
            self.n_classes = self.n_classes + n_new_classes
        
        if self.args.classifier_type=='clip_0_shot':
            #add new classifier with more classes
            self.model.classifiers=self.prepare_classifier(self.classifier_type,self.feature_size, self.n_classes)
        elif self.args.classifier_type in ['nmc', 'slda']:
            clsf = self.prepare_classifier(self.classifier_type,self.feature_size, self.n_classes).to(self.device)
            self.model.expand_classifier(clsf)
        else:
            if task_id>0 and self.args.classifier_type not in sklearn_classifiers:
                if self.args.multihead:    
                    cls = self.prepare_classifier(self.classifier_type,self.feature_size, n_new_classes)[-1].to(self.device)
                    self.model.add_classifier_head(cls)
                elif expand_single_head:
                    clsf = self.prepare_classifier(self.classifier_type,self.feature_size, self.n_classes).to(self.device)
                    self.model.expand_classifier(clsf)
                
                self.model.set_optimizer()

    def reinit_model(self):   
        mapping = self.model.mapping
        self.init_model()
        self.model.mapping = mapping

    def prepare_classifier(self, name, in_dim:int, n_classes:int, encoder=None, args_classifier=None):
        if args_classifier is None:
            args_classifier=self.args_classifier
        if name is None:
            return nn.Identity()
        elif name=='fc':
           classifier = Classifier_nn(in_dim, n_classes, options=args_classifier, device=self.device)
        elif name=='BiT_classifier':
            # if self.args.encoder_name in BiT_models.KNOWN_MODELS.keys():
            #     self.args.flatten_features=0
            #     assert self.args.encoder_name in BiT_models.KNOWN_MODELS.keys()        
            #     classifier = BiT_models.KNOWN_MODELS[self.args.encoder_name](head_size=n_classes, zero_head=True).head
            #     classifier=classifier.to(self.device)
            # else:
            classifier = Classifier_BiT(in_dim, n_classes, options=args_classifier).to(self.device)
        elif name == 'clip_0_shot':
            assert self.classes_text_description is not None
            if encoder is None:
                encoder, _ = self.prepare_encoder(self.args.encoder)   
            text_encodings = encoder.feature_extractor.encode_text(self.classes_text_description.to(self.device))
            encoder = None     
            classifier = CLIPZeroShotClassifier(text_encodings)
        elif name == 'logistic_regression':
            classifier = LogisticRegression_Classifier(solver='lbfgs', max_iter=1000, verbose=1, random_state=0, options=args_classifier)
        elif name == 'random_forrest':
            classifier = RandomForest_Classifier(random_state=0, options=args_classifier)
        elif name == 'knn':
            classifier = KNeighbors_Classifier(options=args_classifier)
        elif name == 'weightnorm':
            classifier = WeightNorm_Classifier(in_dim,n_classes,options=args_classifier, device=self.device)
            classifier=classifier.to(self.device)
        elif name == 'slda':
            classifier = SLDA_Classifier(in_dim,n_classes,options=args_classifier, device=self.device)
            # classifier=classifier.to(self.device)
        elif name=='nmc':
            classifier=NMC_Classifier(in_dim,n_classes,options=args_classifier, device=self.device)
            # classifier=classifier.to(self.device)
        else:
            raise NotImplementedError

        if self.args.multihead and not name=='clip_0_shot':
            if isinstance(classifier, nn.Module):
                classifier = nn.ModuleList([copy.deepcopy(classifier) for _ in range(self.n_tasks)])
                # for cls in classifier:
                    # cls.init_weights()
            else:
                classifier = [classifier for _ in range(self.n_tasks)]
        return classifier

class ModelContainer_ER(ModelContainer): 
    '''
    This class contructs trainable model containing a feature extractor and a classifier as well as an ER buffer.
    '''   
    def __init__(self, args_model: ModelContainer.Options, args_global:ArgsGenerator, args_classifier, n_classes:List[int], device='cuda', ddp=False) -> None:        
        self.args_global = args_global
        classes_text_description=None      
        if 'clip' in args_model.encoder_name:
            classes_text_description = self.get_class_text_encodings()
            classes_text_description=classes_text_description[:n_classes[0]]#.to(device)

        print('ModelContainer_ER',device)
        super().__init__(args_model, args_classifier, n_classes[0], classes_text_description=classes_text_description, 
                        device=device, ddp=ddp, default_transfroms=args_global.dataset.dataset_info.transformations,                   
                        default_transfroms_val=args_global.dataset.dataset_info.transformations_val, weights_path=args_global.weights_path)
        
                            
        self.replay_buffer: continuum.rehearsal.memory.RehearsalMemory = rehearsal.RehearsalMemory(
                                memory_size=args_global.er_size_total,
                                herding_method="random",
                                fixed_memory=True,
                                nb_total_classes=args_global.dataset.dataset_info.n_classes
                            )
        self.results = []

    def get_class_text_encodings(self):  
        if self.args_global.dataset.dataset_info.taxonomy is not None:
            classes_text_description= torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.args_global.dataset.dataset_info.taxonomy])
            if classes_text_description is not None:    
                classes_text_description.requires_grad=False
        else:
            classes_text_description=None
        return classes_text_description

    def init_model(self):
        super().init_model() 
        if self.args_global.regime=="latent_ER" and self.args.classifier_type=='clip_0_shot':
            self.model.feature_extractor=nn.Identity()
        if self.model.optimizer is None:
            self.model.set_optimizer()
        if self.args_global.regime=='latent_ER': 
            self.transforms=None
            self.transforms_val=None

    def set_mapping(self,mapping):
        self.model.mapping=mapping
    
    def ready_for_new_task(self, task_id, new_classes, expand_single_head=True):
        '''
        On task switch routine.
        '''
        n_new_classes = len(new_classes)
        # self.model.update_mapping(task_id,new_classes)
        if task_id>0:
            classes_text_description = self.get_class_text_encodings()
            #take only embedings of classes until current task (current n_classes)
            if classes_text_description is not None:
                self.classes_text_description = classes_text_description[:self.n_classes+n_new_classes]#.to(device)
            return super().ready_for_new_task(task_id, n_new_classes, expand_single_head)

    def add_to_buffer(self,samples:Tuple):
        return self.replay_buffer.add(*samples,z=None)

    def get_replay_samples(self):
        return self.replay_buffer.get()
    
    def get_model_for_training(self, task_id:int) -> Tuple[ModelContainer.Options, ClassifierOptions, TorchModuleWrapper]:
        '''
        Generates arguments and a model for per task cross validation.
        '''
        if not self.args_global.task_level_cv or task_id>0 and self.args_global.keep_best_params_after_first_task:
            yield None, None, self.model
        
        if self.args_global.task_level_cv and (not self.args_global.keep_best_params_after_first_task or task_id==0):
            for args, args_classifier in self.CV_args():

                model, _ = self.create_model(args, args_classifier)
                model.load_state_dict(self.model.state_dict())

                if self.args_global.regime=="latent_ER" and self.args.classifier_type=='clip_0_shot':
                    model.feature_extractor=nn.Identity()
                model.set_optimizer()
                yield args, args_classifier, model
            
    def reset_args(self, args_model:ModelContainer.Options, args_classifier:ClassifierOptions):
        '''
        Sets the arguments of the model container and classifier to the given values (used for per task CV).
        '''
        if args_model is not None:
            self.args = copy.copy(args_model)
        if args_classifier is not None:
            self.args_classifier = copy.copy(args_classifier)

from collections import namedtuple
from dataclasses import dataclass
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
from torch.nn.modules.pooling import MaxPool2d 
# import vissl
import wandb
from PIL import Image    
from functools import partial  
from fvcore.nn import parameter_count_table
from torchvision.models.resnet import resnet18
from simple_parsing import choice
from torchvision.transforms.transforms import Lambda, ToPILImage
from . import BiT_models, resnets_cifar, models_pt_cifar
from torchvision import models as torch_models
from torchvision import transforms   
# from .MIIL.factory import create_model as create_model_MIIL
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification
from .module_wrapper import TorchModuleWrapper
from .helper import torch_NN, bn_eval, getKeysByValue
from .classifiers import Classifier_nn

project_home = f"{os.environ.get('LARGE_CL_HOME')}" if "LARGE_CL_HOME" in os.environ else "."
_models_path = f"{os.environ.get('SCRATCH')}/Weights/" if "SCRATCH" in os.environ else f"{project_home}/Weights"

@dataclass  
class PreparedModel:
    encoder: nn.Module
    latent_dim: object
    transformation: object
    transformation_val: object

@dataclass
class EncoderTuple:
    partial_encoder: Callable
    pretrained_options: List
    info_string: str
    pretrain_dataset: str

# EncoderTuple = namedtuple('EncoderTuple',['partial_encoder','pretrained_options','info_string', 'pretrain_dataset'])
               
def prepare_BIT_model_timm(name, pretrained=True, fix_batchnorms_encoder=True, models_path=_models_path, device='cuda', hidden_layer=None, *args, **kwargs)->PreparedModel:  
        # model = BiT_models.KNOWN_MODELS[name](head_size=10, zero_head=True)
        import urllib 
        from PIL import Image  
        from timm.data import resolve_data_config  
        from timm.data.transforms_factory import create_transform

        model = timm.create_model(name, pretrained=True, num_classes=0)
        print("encoder",model)
        model.to(device)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)  
        image = torch.Tensor(torch.ones(5,3,32,32)).to(device)
        if hidden_layer is not None and hidden_layer<=4:
            model.stages=model.stages[:hidden_layer]
            model.norm=nn.Identity()

        latent_dim = np.prod(model(image).shape[1:])                
        model = TorchModuleWrapper(model, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder).to(device) 
        return PreparedModel(model, latent_dim, transform.transforms, transform.transforms)

def prepare_model_timm(name, pretrained=True, fix_batchnorms_encoder=True, models_path=_models_path, device='cuda', *args, **kwargs)->PreparedModel:  
        # model = BiT_models.KNOWN_MODELS[name](head_size=10, zero_head=True)
        import urllib 
        from PIL import Image
        from timm.data import resolve_data_config  
        from timm.data.transforms_factory import create_transform

        model = timm.create_model(name, pretrained=True, num_classes=0)
        print("encoder",model)
        model.to(device)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)       
        image = transform(transforms.ToPILImage()(torch.Tensor(torch.ones(3,32,32)))).unsqueeze(0).to(device)
        out = model(image)
        if 'deit' in name and isinstance(out, Tuple):
            latent_dim = np.prod(out[0].shape[1:])
            #classifier will be built on the average of both features
            def forward(model, x): 
                if model.training:
                    out= model(x)
                    out = (out[0]+out[1])/2       
                    return out.unsqueeze(0)       
                else:
                    return model(x)
            model = TorchModuleWrapper(model, function=forward, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder).to(device) 
        else:
            latent_dim = np.prod(out.shape[1:])            
            model = TorchModuleWrapper(model, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder).to(device) 
        return PreparedModel(model, latent_dim, transform.transforms, transform.transforms)


def prepare_BIT_model(name, pretrained=True, fix_batchnorms_encoder=True, models_path=_models_path, device='cuda',  *args, **kwargs)->PreparedModel:  
        model = BiT_models.KNOWN_MODELS[name](head_size=10, zero_head=True)

        if pretrained: 
            if not os.path.isfile(f"{models_path}/{name}.npz".replace("//","/")):
                print(f"Downloading model from {name}.npz to {models_path}")        
                #download to a temp location file, to prevent from concurent downloads if many runs run at the same time
                n = wandb.util.generate_id() #np.random.randint(0,1000000)
                tmp_folder = f"{models_path}/tmp_{n}/"
                os.makedirs(tmp_folder)
                os.system(f"wget https://storage.googleapis.com/bit_models/{name}.npz -P {tmp_folder}")
                model.load_from(np.load(f"{tmp_folder}/{name}.npz".replace("//","/")))
                #move to permanent location and remove temp folder
                if not os.path.isfile(f"{models_path}/{name}.npz".replace("//","/")):
                    os.system(f"mv {tmp_folder}/{name}.npz {models_path}/{name}.npz")
                os.system(f"rm -r {tmp_folder}")
            else:    
                model.load_from(np.load(f"{models_path}/{name}.npz".replace("//","/")))
        
        #from https://github.com/google-research/big_transfer/blob/49afe42338b62af9fbe18f0258197a33ee578a6b/bit_hyperrule.py#L30
        def get_resolution(original_resolution):
            # """Takes (H,W) and returns (precrop, crop)."""
            # area = original_resolution[0] * original_resolution[1]
            return (100,100) #(512, 480) #(160, 128) if area < 96*96 else (512, 480)
        precrop, crop = get_resolution(None)
        transformation = [
                    transforms.Resize((precrop, precrop)),
                    transforms.RandomCrop((crop, crop)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        latent_dim = list(model.children())[-1][-1].weight.shape[1]
        model.head.conv=nn.Identity()
        features_exctractor = nn.Sequential(*list(model.children()))
        # classifier = None
        # if n_classes is not None:
        #     classifier = nn.ModuleList([BiT_models.KNOWN_MODELS[name](head_size=n_cls, zero_head=True).head for n_cls in n_classes]).to(self.device)
            
        # if self.debug:
        #     features_exctractor = nn.Sequential(*[nn.Flatten(1), nn.Linear(self.args.in_size*self.args.in_size*3,latent_dim)]).to(self.device)
        #     if n_classes is not None:
        #         classifier = nn.ModuleList([nn.Linear(latent_dim,n_cls) for n_cls in n_classes]).to(self.device)
        # if classifier is None:
        #     return TorchModuleWrapper(features_exctractor, flatten=False)
        model = TorchModuleWrapper(features_exctractor, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder).to(device) 
        return PreparedModel(model, latent_dim, transformation, transformation)

def prepare_vit_model(encoder_name, pretrained=True, vit_out_layer='last_hidden_state', fix_batchnorms_encoder=True, debug=False, device='cuda', hidden_layer=None, *args, **kwargs)->PreparedModel:
    if encoder_name=='ViT-B/32':    
        feature_extractor_vit = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-224-in21k')
        # if pretrained:
        # model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-224-in21k')  
    elif encoder_name=='ViT-B/16':
        feature_extractor_vit = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        # if pretrained:
        # model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')     
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')     
    model.classifier = nn.Identity()
    # if debug:    
    #     _transforms = transforms.ToTensor()
    #     _transforms_val = transforms.ToTensor()
    # else:
    #     _transforms = lambda x: np.array(x).transpose(2,0,1).astype(np.uint8)
    #     _transforms_val = lambda x: np.array(x).transpose(2,0,1).astype(np.uint8)     



    def forward(model, x): 
        return model(x).logits

    transformation = [Lambda(lambda x: feature_extractor_vit(x, return_tensors="pt").pixel_values[0])] # (images=[im.cpu() for im in x], return_tensors="pt"))]

    if hidden_layer is not None and hidden_layer<12:
        model.vit.encoder.layer=model.vit.encoder.layer[:hidden_layer]

    image = torch.Tensor(torch.ones(feature_extractor_vit.size,feature_extractor_vit.size,3)).unsqueeze(0).to(device)
    image = feature_extractor_vit(images=[im.cpu() for im in image], return_tensors="pt")
    image = image.to(device)
    model.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feature_size = model(image.pixel_values).logits.shape[1:][0]
    if was_training:
        model.train()
    # if self.debug:
    #     model = TorchModuleWrapper(self.args, nn.Sequential(*[nn.Flatten(1), nn.Linear(32*32*3,512)]))
    model = TorchModuleWrapper(model, function=forward, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder, flatten_features=False).to(device) #
    model.to(device)
    return PreparedModel(model, feature_size, transformation, transformation)

def prepare_dino_model(encoder_name, fix_batchnorms_encoder, device='cuda', *args, **kwargs)->PreparedModel:
    vits = torch.hub.load('facebookresearch/dino:main', encoder_name).to(device)
    image = torch.Tensor(torch.ones(5,3,32,32)).to(device)
    latent_dim = np.prod(vits(image).shape[1:])
    return PreparedModel(TorchModuleWrapper(vits, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder), latent_dim, None, None)
      
def prepare_dino_model_fugging_face(encoder_name, fix_batchnorms_encoder, device='cuda', *args, **kwargs)->PreparedModel:
    # if encoder_name=='ViT-B/32':    
    feature_extractor_vit = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
    model = ViTModel.from_pretrained('facebook/dino-vitb8')
    # model.classifier = nn.Identity()
    def forward(model, x):     
        outputs = model(x)  
        last_hidden_states = outputs.pooler_output  
        return last_hidden_states

    transformation = [Lambda(lambda x: feature_extractor_vit(x, return_tensors="pt").pixel_values[0])] # (images=[im.cpu() for im in x], return_tensors="pt"))]

    image = torch.Tensor(torch.ones(feature_extractor_vit.size,feature_extractor_vit.size,3)).unsqueeze(0).to(device)
    image = feature_extractor_vit(images=[im.cpu() for im in image], return_tensors="pt")
    image = image.to(device)
    model.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad(): 
        feature_size = model(image.pixel_values).pooler_output.shape[1:][0]
    if was_training:
        model.train()
    model = TorchModuleWrapper(model, function=forward, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder, flatten_features=False).to(device) #
    model.to(device)
    return PreparedModel(model, feature_size, transformation, transformation)


def prepare_encoder_torch(model:nn.Module, pretrained=True, fix_batchnorms_encoder=True, *args, **kwargs)->PreparedModel:
    model = model(pretrained=pretrained)     
    latent_dim = model.classifier[-1].in_features   
    model.classifier[-1] = nn.Identity() 
    features_exctractor = TorchModuleWrapper(model, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder)
    return PreparedModel(features_exctractor,latent_dim, None, None)
    
def prepare_encoder_torch_resnet(resnet:nn.Module, pretrained=True, device='cuda', fix_batchnorms_encoder=True, *args, **kwargs)->PreparedModel:
    model = resnet(pretrained=pretrained)
    latent_dim = list(model.children())[-1].in_features 
    image = torch.Tensor(torch.ones(5,3,32,32)).to(device)
    features_exctractor = TorchModuleWrapper(nn.Sequential(*list(model.children())[:-1]), keep_bn_in_eval_after_freeze=fix_batchnorms_encoder)
    return PreparedModel(features_exctractor,latent_dim, None, None)
          
def prepare_clip_model(encoder_name, device='cuda', fix_batchnorms_encoder=True, *args, **kwargs)->PreparedModel:
    model, clip_transforms = clip.load(encoder_name, device=device)
    # image = clip_transforms(Image.open(f"{project_home}/Utils/clip/CLIP.png")).unsqueeze(0).to(device)
    image = clip_transforms(transforms.ToPILImage()(torch.Tensor(torch.ones(3,224,224)))).unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():            
        latent_dim = model.encode_image(image).shape[1]
    if was_training:
        model.train()
    # def forward(model,inp):
    #     return m.encode_image(inp)
    # model = TorchModuleWrapper(model, function=lambda m,inp: m.encode_image(inp), keep_bn_in_eval_after_freeze=self.args.fix_batchnorms_encoder)
    model = TorchModuleWrapper(model, function=model.encode_image, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder)
    return PreparedModel(model, latent_dim, clip_transforms.transforms, clip_transforms.transforms)

def prepare_resnet_cifar(encoder_name, pretrained=False, fix_batchnorms_encoder=False, device='cuda', *args, **kwargs)->PreparedModel:
    encoder = resnets_cifar.__dict__[encoder_name]()
    feature_size = list(encoder.children())[-1].in_features    
    encoder.linear = nn.Identity()
    encoder = TorchModuleWrapper(encoder, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder).to(device)
    return PreparedModel(encoder, feature_size, None, None)

def prepare_pt_cifar_model(encoder_name, pretrained=False, fix_batchnorms_encoder=False, *args, **kwargs)->PreparedModel: 
    if 'VGG' in encoder_name:        
        encoder = models_pt_cifar.__dict__[encoder_name]('VGG16')
    else:
        encoder = models_pt_cifar.__dict__[encoder_name]()
    latent_dim = list(encoder.children())[-1].in_features    
    encoder = list(encoder.children())[:-1] + [torch.nn.AvgPool2d(4)]                     
    features_exctractor = TorchModuleWrapper(nn.Sequential(*encoder), keep_bn_in_eval_after_freeze=fix_batchnorms_encoder)
    return PreparedModel(features_exctractor, latent_dim, None, None)
      
def prepare_fc_encoder(in_size, feature_size, *args, **kwargs):
    # assert self.args.in_size is not None
    # assert not self.args.pretrained_encoder
    # encoder = self.prepare_classifier(encoder_name, self.args.in_size, n_classes=[feature_size])
    encoder = Classifier_nn(in_size, feature_size)
    return PreparedModel(encoder, feature_size, None, None)

def prepare_MIIL_timm(encoder_name, pretrained, device='cuda', *args, **kwargs):
    model = timm.create_model(encoder_name, pretrained=pretrained)
    model.to(device)
    return PreparedModel(model, 10, None, None)

# def prepare_MIIL(encoder_name, fix_batchnorms_encoder, device=device, models_path=_models_path, *args, **kwargs):
#     if encoder_name=='resnet50_MIIL':
#         encoder_name='resnet50'
#         file_name = 'resnet50_miil_21k'
#     else:
#         raise NotImplementedError
#     @dataclass
#     class Args:  
#         model_name:str
#         num_classes: int
#         model_path:str = f"{models_path}/{file_name}.pth".replace("//","/")

#     if not os.path.isfile(f"{models_path}/{file_name}.pth".replace("//","/")):
#         print(f"Downloading model from {file_name}.npz to {models_path}")        
#         #download to a temp location file, to prevent from concurent downloads if many runs run at the same time
#         n = wandb.util.generate_id() #np.random.randint(0,1000000)
#         tmp_folder = f"{models_path}/tmp_{n}/"
#         os.makedirs(tmp_folder)
#         os.system(f"wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth -P {tmp_folder}")
#         model = create_model_MIIL(Args(model_name=encoder_name, num_classes=10, model_path=f"{tmp_folder}/{file_name}.pth"))
        
#         #move to permanent location and remove temp folder
#         if not os.path.isfile(f"{models_path}/{file_name}.npz".replace("//","/")):
#             os.system(f"mv {tmp_folder}/{file_name}.npz {models_path}/{file_name}.npz")
#         os.system(f"rm -r {tmp_folder}")

#         # print(f"Downloading model from {file_name}.pth".replace("//","/"))                                 
#         # os.system(f"wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth -P {models_path}")
#     else:
#         model = create_model_MIIL(Args(model_name=encoder_name, num_classes=10))
#     model.to(device)
#     latent_dim = list(model.children())[-1].in_features    
#     encoder = list(model.children())[:-1]         
#     encoder = TorchModuleWrapper(nn.Sequential(*encoder), keep_bn_in_eval_after_freeze=fix_batchnorms_encoder)
#     return PreparedModel(encoder, latent_dim, None, None)

# def prepare_vissl(encoder_name, link, config, models_path=_models_path, fix_batchnorms_encoder=True, device='cuda', *args, **kwargs):
#     from omegaconf import OmegaConf
#     from vissl.utils.hydra_config import AttrDict    
#     from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
#     file_name=f'{encoder_name}.torch'
#     path=f"{models_path}/{file_name}".replace("//","/")
#     if not os.path.isfile(path):
#         os.system(f"wget -q -O {path} {link}")
#     # Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
#     # All other options override the simclr_8node_resnet.yaml config.

#     cfg = [
#             f'config=pretrain/simclr/models/{config}',
#             f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={path}', # Specify path for the model weights.
#             'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
#             'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
#             'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
#             'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
#             'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]' # Extract only the res5avg features.
#           ]

#     # Compose the hydra configuration.
#     cfg = compose_hydra_configuration(cfg)
#     # Convert to AttrDict. This method will also infer certain config options
#     # and validate the config is valid.
#     _, cfg = convert_to_attrdict(cfg)
#     from vissl.models import build_model
#     model = build_model(cfg.MODEL, cfg.OPTIMIZER)   
#     from classy_vision.generic.util import load_checkpoint
#     from vissl.utils.checkpoint import init_model_from_consolidated_weights

#     # Load the checkpoint weights.
#     weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)


#     # Initializei the model with the simclr model weights.
#     init_model_from_consolidated_weights(
#         config=cfg,
#         model=model,
#         state_dict=weights,
#         state_dict_key_name="classy_state_dict",
#         skip_layers=[],  # Use this if you do not want to load all layers
#     )
#     # transfroms = transforms.Compose([
#     #   transforms.CenterCrop(224),
#     #   transforms.ToTensor(),
#     # ])

#     print("Weights have loaded")  
#     image = torch.Tensor(torch.ones(5,3,32,32)).to(device)
#     model.to(device)
#     feature_size=np.prod(model(image)[0].shape[1:])
#     def forward(model, x):
#         return model(x)[0]
#     model=TorchModuleWrapper(model, function=forward, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder, flatten_features=False).to(device) #
#     return PreparedModel(model, feature_size, None, None)
      
def prepare_swsl(encoder_name, fix_batchnorms_encoder=False, device='cuda', *args, **kwargs):
    import urllib
    from PIL import Image
    from timm.data import resolve_data_config  
    from timm.data.transforms_factory import create_transform
    model = timm.create_model(encoder_name, pretrained=True, num_classes=0)
    model.to(device)
    # model.eval()    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    image = torch.Tensor(torch.ones(5,3,32,32)).to(device)
    latent_dim = np.prod(model(image).shape[1:])
    return PreparedModel(TorchModuleWrapper(model, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder), latent_dim, transform.transforms, transform.transforms)

def prepare_efficient_net(name, fix_batchnorms_encoder=False, device='cuda', *args, **kwargs):
    model = timm.create_model(name, pretrained=True)
    model.classifier=nn.Identity()
    model.to(device)
    image = torch.Tensor(torch.ones(5,3,32,32)).to(device)
    latent_dim = np.prod(model(image).shape[1:])
    from timm.data.transforms_factory import create_transform
    from timm.data import resolve_data_config
    config = resolve_data_config({}, model=model)  
    transform = create_transform(**config).transforms
    return PreparedModel(TorchModuleWrapper(model, keep_bn_in_eval_after_freeze=fix_batchnorms_encoder), latent_dim, transform, transform)

def prepare_cnn(encoder_name, device='cuda', input_shape=32, fix_batchnorms_encoder=True, *args, **kwargs)->PreparedModel:
    # Assemble the network model
    model = nn.ModuleList() #nn.Sequential()        

    class Model(nn.Module):
        def __init__(self,input_shape):
            super().__init__()
            model = nn.ModuleList()
            model.append(nn.Conv2d(3, 32, (3, 3), padding='same'))#,   
                                #input_shape=[input_shape,input_shape])) #training_datasets[0][0].shape[1:]))
            model.append(nn.ReLU())
            model.append(nn.Conv2d(32,32, (3, 3)))
            model.append(nn.ReLU())
            model.append(nn.MaxPool2d((2, 2)))
            model.append(nn.Dropout(0.25))
            model.append(nn.Conv2d(32,64, (3, 3), padding='same'))
            model.append(nn.ReLU())
            model.append(nn.Conv2d(64,64, (3, 3)))
            model.append(nn.ReLU())
            model.append(nn.MaxPool2d(kernel_size=(2, 2)))
            model.append(nn.Dropout(0.25))
            model.append(nn.Flatten())
            image = torch.Tensor(torch.ones(5,3,input_shape,input_shape)).to(device)
            in_sh=nn.Sequential(*model).to(device)(image).shape[-1]
            model.append(nn.Linear(in_sh,512))
            # model.append(nn.ReLU())
            self.model=model
        def forward(self,x):
            for l in self.model:
                x=l(x)
            return x

    # image = torch.Tensor(torch.ones(5,3,input_shape,input_shape)).to(device)
    # in_sh=nn.Sequential(*model).to(device)(image).shape[-1]
    # model.append(nn.Linear(in_sh,512))
    # model.append(nn.ReLU())
    # model.append(nn.Dropout(0.5))
    # model.add(Dense(nb_classes))
    # model.append(nn.Linear(nb_classes, kernel_initializer='zero', activation=masked_softmax))
    model=Model(input_shape)
    return PreparedModel(TorchModuleWrapper(nn.Sequential(model), keep_bn_in_eval_after_freeze=fix_batchnorms_encoder), 512, None, None) 


encoders={   
    "None": None, 
    "fc": EncoderTuple(partial(prepare_fc_encoder),pretrained_options=[False], info_string='Just a fully connected netork', pretrain_dataset='None'),
    #models from here: https://github.com/kuangliu/pytorch-cifar
    "pt_ResNet18": EncoderTuple(partial(prepare_pt_cifar_model,'ResNet18'),pretrained_options=[False], info_string='ResNet18 from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_VGG16": EncoderTuple(partial(prepare_pt_cifar_model, 'VGG'),pretrained_options=[False], info_string='VGG from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_ResNet50": EncoderTuple(partial(prepare_pt_cifar_model,'ResNet50'),pretrained_options=[False], info_string='ResNet50 from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_ResNet101": EncoderTuple(partial(prepare_pt_cifar_model,'ResNet101'),pretrained_options=[False], info_string='ResNet101 from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_ResNet152": EncoderTuple(partial(prepare_pt_cifar_model,'ResNet152'),pretrained_options=[False], info_string='ResNet152 from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_DenseNet121": EncoderTuple(partial(prepare_pt_cifar_model,'DenseNet121'),pretrained_options=[False], info_string='DenseNet121 from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_DenseNet161": EncoderTuple(partial(prepare_pt_cifar_model,'DenseNet161'),pretrained_options=[False], info_string='DenseNet161 from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_DenseNet201": EncoderTuple(partial(prepare_pt_cifar_model,'DenseNet201'),pretrained_options=[False], info_string='DenseNet201 from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    "pt_SimpleDLA": EncoderTuple(partial(prepare_pt_cifar_model,'SimpleDLA'),pretrained_options=[False], info_string='SimpleDLA from here: https://github.com/kuangliu/pytorch-cifar', pretrain_dataset='None'),
    ###############################
    #from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    'resnet20_cifar': EncoderTuple(partial(prepare_resnet_cifar, 'resnet20'),pretrained_options=[False], info_string='resnet20 from here: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py', pretrain_dataset='None'),
    'resnet32_cifar': EncoderTuple(partial(prepare_resnet_cifar, 'resnet32'),pretrained_options=[False], info_string='resnet32 from here: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py', pretrain_dataset='None'),
    'resnet44_cifar': EncoderTuple(partial(prepare_resnet_cifar, 'resnet44'),pretrained_options=[False], info_string='resnet44 from here: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py', pretrain_dataset='None'),
    'resnet56_cifar': EncoderTuple(partial(prepare_resnet_cifar, 'resnet56'),pretrained_options=[False], info_string='resnet56 from here: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py', pretrain_dataset='None'),
    'resnet110_cifar': EncoderTuple(partial(prepare_resnet_cifar, 'resnet110'),pretrained_options=[False], info_string='resnet110 from here: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py', pretrain_dataset='None'),
    'resnet1202_cifar': EncoderTuple(partial(prepare_resnet_cifar, 'resnet1202'),pretrained_options=[False], info_string='resnet1202 from here: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py', pretrain_dataset='None'),
    ################################
    #models from pytorch  
    'cnn':EncoderTuple(partial(prepare_cnn,''), pretrained_options=[True, False], info_string='resnet18 from torch, can be pretrained on ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnet18':EncoderTuple(partial(prepare_encoder_torch_resnet,torch_models.resnet18),pretrained_options=[True, False], info_string='resnet18 from torch, can be pretrained on ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnet34':EncoderTuple(partial(prepare_encoder_torch_resnet,torch_models.resnet34),pretrained_options=[True, False], info_string='resnet34 from torch, can be pretrained on ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnet50':EncoderTuple(partial(prepare_encoder_torch_resnet,torch_models.resnet50),pretrained_options=[True, False], info_string='resnet50 from torch, can be pretrained on ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnet101':EncoderTuple(partial(prepare_encoder_torch_resnet,torch_models.resnet101),pretrained_options=[True, False], info_string='resnet101 from torch, can be pretrained on ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnet152':EncoderTuple(partial(prepare_encoder_torch_resnet,torch_models.resnet152),pretrained_options=[True, False], info_string='resnet152 from torch, can be pretrained on ImageNet1K', pretrain_dataset='ImageNet1K'),
    ################################
    # CLIP models 
    'RN50_clip': EncoderTuple(partial(prepare_clip_model,'RN50'),pretrained_options=[True, False], info_string='ResNet50 pretrained with clip on the clip dataset', pretrain_dataset='clip'),
    'RN101_clip': EncoderTuple(partial(prepare_clip_model,'RN101'),pretrained_options=[True, False], info_string='ResNet101 pretrained with clip on the clip dataset', pretrain_dataset='clip'),
    'RN50x4_clip': EncoderTuple(partial(prepare_clip_model,'RN50x4'),pretrained_options=[True, False], info_string='ResNet50x4 pretrained with clip on the clip dataset', pretrain_dataset='clip'),
    'RN50x16_clip': EncoderTuple(partial(prepare_clip_model,'RN50x16'),pretrained_options=[True, False], info_string='ResNet50x16 pretrained with clip on the clip dataset', pretrain_dataset='clip'),
    'ViT-B/32_clip': EncoderTuple(partial(prepare_clip_model,'ViT-B/32'),pretrained_options=[True, False], info_string='ViT-B/32 pretrained with clip on the clip dataset', pretrain_dataset='clip'),
    'ViT-B/16_clip': EncoderTuple(partial(prepare_clip_model,'ViT-B/16'),pretrained_options=[True, False], info_string='ViT-B/16 pretrained with clip on the clip dataset', pretrain_dataset='clip'),
    'ViT-L/14_clip': EncoderTuple(partial(prepare_clip_model,'ViT-L/14'),pretrained_options=[True, False], info_string='ViT-B/16 pretrained with clip on the clip dataset', pretrain_dataset='clip'),
    'RN50x64_clip': EncoderTuple(partial(prepare_clip_model,'RN50x64'),pretrained_options=[True, False], info_string='ViT-B/16 pretrained with clip on the clip dataset', pretrain_dataset='clip'),

    
    ######### 
    #Vision transformers from https://huggingface.co/transformers/model_doc/vit.html
    'ViT-B/32': EncoderTuple(partial(prepare_vit_model,'ViT-B/32'),pretrained_options=[True], info_string='ViT-B/32 from https://huggingface.co/transformers/model_doc/vit.html can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'ViT-B/16': EncoderTuple(partial(prepare_vit_model,'ViT-B/16'),pretrained_options=[True], info_string='ViT-B/16 from https://huggingface.co/transformers/model_doc/vit.html can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    # Models from Big Transfer Paper https://github.com/google-research/big_transfer
    'BiT-S-R50x1': EncoderTuple(partial(prepare_BIT_model,'BiT-S-R50x1'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'BiT-S-R50x3': EncoderTuple(partial(prepare_BIT_model,'BiT-S-R50x3'),pretrained_options=[True, False], info_string='ResNet50x3 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'BiT-S-R101x1': EncoderTuple(partial(prepare_BIT_model,'BiT-S-R101x1'),pretrained_options=[True, False], info_string='ResNet101x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'BiT-S-R101x3': EncoderTuple(partial(prepare_BIT_model,'BiT-S-R101x3'),pretrained_options=[True, False], info_string='ResNet101x3 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'BiT-S-R152x2': EncoderTuple(partial(prepare_BIT_model,'BiT-S-R152x2'),pretrained_options=[True, False], info_string='ResNet152x2 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'BiT-S-R152x4': EncoderTuple(partial(prepare_BIT_model,'BiT-S-R152x4'),pretrained_options=[True, False], info_string='ResNet152x4 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'BiT-M-R50x1': EncoderTuple(partial(prepare_BIT_model,'BiT-M-R50x1'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'BiT-M-R50x3': EncoderTuple(partial(prepare_BIT_model,'BiT-M-R50x3'),pretrained_options=[True, False], info_string='ResNet50x3 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'BiT-M-R101x1': EncoderTuple(partial(prepare_BIT_model,'BiT-M-R101x1'),pretrained_options=[True, False], info_string='ResNet101x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'BiT-M-R101x3': EncoderTuple(partial(prepare_BIT_model,'BiT-M-R101x3'),pretrained_options=[True, False], info_string='ResNet101x3 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'BiT-M-R152x2': EncoderTuple(partial(prepare_BIT_model,'BiT-M-R152x2'),pretrained_options=[True, False], info_string='ResNet152x2 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'BiT-M-R152x4': EncoderTuple(partial(prepare_BIT_model,'BiT-M-R152x4'),pretrained_options=[True, False], info_string='ResNet152x4 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
      
    'tf_efficientnet_l2_ns_475': EncoderTuple(partial(prepare_model_timm,'tf_efficientnet_l2_ns_475'),pretrained_options=[True, False], info_string='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models', pretrain_dataset='ImageNet1K'),
    'deit_base_distilled_patch16_224': EncoderTuple(partial(prepare_model_timm,'deit_base_distilled_patch16_224'),pretrained_options=[True, False], info_string='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models', pretrain_dataset='ImageNet1K'),
    'resnet50_timm': EncoderTuple(partial(prepare_model_timm,'resnet50'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'ssl_resnet50': EncoderTuple(partial(prepare_model_timm,'ssl_resnet50'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    
    # vit_base_patch16_224_in21k
    'resnetv2_50x1_bitm': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_50x1_bitm'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnetv2_50x1_bitm_in21k': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_50x1_bitm_in21k'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnetv2_101x1_bitm_in21k': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_101x1_bitm_in21k'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnetv2_101x3_bitm_in21k': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_101x3_bitm_in21k'),pretrained_options=[True, False], info_string='ResNet50x3 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnetv2_152x2_bitm_in21k': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_152x2_bitm_in21k'),pretrained_options=[True, False], info_string='ResNet101x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnetv2_101x3_bitm_in21k': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_101x3_bitm_in21k'),pretrained_options=[True, False], info_string='ResNet101x3 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnetv2_152x4_bitm_in21k': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_152x4_bitm_in21k'),pretrained_options=[True, False], info_string='ResNet152x4 from https://github.com/google-research/big_transfer can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'resnetv2_50x1_bit_distilled': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_50x1_bit_distilled'),pretrained_options=[True, False], info_string='ResNet50x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'resnetv2_152x2_bit_teacher': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_152x2_bit_teacher'),pretrained_options=[True, False], info_string='ResNet50x3 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    'resnetv2_152x2_bit_teacher_384': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_152x2_bit_teacher_384'),pretrained_options=[True, False], info_string='ResNet101x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
    # 'resnetv2_50d_evos': EncoderTuple(partial(prepare_BIT_model_timm,'resnetv2_50d_evos'),pretrained_options=[True, False], info_string='ResNet101x1 from https://github.com/google-research/big_transfer can be pretrained with ImageNet21K', pretrain_dataset='ImageNet21K'),
        
    #dino models
    'dino_vits16': EncoderTuple(partial(prepare_dino_model,'dino_vits16'),pretrained_options=[True], info_string='ViT-B/16 from https://github.com/facebookresearch/dino can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'dino_vits8': EncoderTuple(partial(prepare_dino_model,'dino_vits8'),pretrained_options=[True], info_string='ViT-B/8 from https://github.com/facebookresearch/dino can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'dino_vitb16': EncoderTuple(partial(prepare_dino_model,'dino_vitb16'),pretrained_options=[True], info_string='ViT-B/16 from https://github.com/facebookresearch/dino can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'dino_vitb8': EncoderTuple(partial(prepare_dino_model,'dino_vitb8'),pretrained_options=[True], info_string='ViT-B/8 from https://github.com/facebookresearch/dino can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'dino_vitb8_hf': EncoderTuple(partial(prepare_dino_model_fugging_face,'dino_vitb8'),pretrained_options=[True], info_string='ViT-B/8 from https://github.com/facebookresearch/dino can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    'dino_resnet50': EncoderTuple(partial(prepare_dino_model,'dino_resnet50'),pretrained_options=[True], info_string='REsNet50 from https://github.com/facebookresearch/dino can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    #swsl https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    'swsl_resnext101_32x16d': EncoderTuple(partial(prepare_swsl,'swsl_resnext101_32x16d'),pretrained_options=[True], info_string='REsNet from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models https://rwightman.github.io/pytorch-image-models/models/swsl-resnet/ pretrained in sem-weakly way on 940 million public images', pretrain_dataset='940M images'),
    'efficient_net_nosy_teacher': EncoderTuple(partial(prepare_efficient_net,'tf_efficientnet_b0_ns'),pretrained_options=[True], info_string='Eficient net from noisy student pretraining https://rwightman.github.io/pytorch-image-models/models/noisy-student/', pretrain_dataset='300M milion'),
    'efficient_net_nosy_teacher_b7': EncoderTuple(partial(prepare_efficient_net, 'tf_efficientnet_b7_ns'),pretrained_options=[True], info_string='Eficient net from noisy student pretraining https://rwightman.github.io/pytorch-image-models/models/noisy-student/', pretrain_dataset='300M milion'),
    'efficient_net_nosy_teacher_b6': EncoderTuple(partial(prepare_efficient_net, 'tf_efficientnet_b6_ns'),pretrained_options=[True], info_string='Eficient net from noisy student pretraining https://rwightman.github.io/pytorch-image-models/models/noisy-student/', pretrain_dataset='300M milion'),
    # 'efficient_net_nosy_teacher': EncoderTuple(partial(prepare_efficient_net, 'tf_efficientnet_b5_ns'),pretrained_options=[True], info_string='Eficient net from noisy student pretraining https://rwightman.github.io/pytorch-image-models/models/noisy-student/', pretrain_dataset='300M milion'),
      
    #models from https://github.com/Alibaba-MIIL/ImageNet21K
    #21K:
    # 'tresnet_l': EncoderTuple(partial(prepare_MIIL,'tresnet_l'),pretrained_options=[True], info_string='tresnet_l from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet21K'),
    # 'ofa_flops_595m_s': EncoderTuple(partial(prepare_MIIL,'ofa_flops_595m_s'),pretrained_options=[True], info_string='ofa_flops_595m_s from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet21K'),
    # 'resnet50_MIIL': EncoderTuple(partial(prepare_MIIL,'resnet50'),pretrained_options=[True], info_string='resnet50_MIIL from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet21K'),
    # 'mobilenetv3_large_100_miil_in21k': EncoderTuple(partial(prepare_MIIL_timm,'mobilenetv3_large_100_miil_in21k'),pretrained_options=[True, False], info_string='mobilenetv3_large_100_miil_in21k from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet21K'),
    # 'tresnet_m_miil_in21k': EncoderTuple(partial(prepare_MIIL_timm,'mobilenetv3_large_100_miil_in21k'),pretrained_options=[True, False], info_string='tresnet_m_miil_in21k from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet21K'),
    # 'vit_base_patch16_224_miil_in21k': EncoderTuple(partial(prepare_MIIL_timm,'mobilenetv3_large_100_miil_in21k'),pretrained_options=[True, False], info_string='vit_base_patch16_224_miil_in21k from https://github.com/Alibaba-MIIL/ImageNet21Kcan be pretrained with ImageNet1K', pretrain_dataset='ImageNet21K'),
    # 'mixer_b16_224_miil_in21k': EncoderTuple(partial(prepare_MIIL_timm,'mobilenetv3_large_100_miil_in21k'),pretrained_options=[True, False], info_string='mixer_b16_224_miil_in21k from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    # #1K:
    # 'mobilenetv3_large_100_miil': EncoderTuple(partial(prepare_MIIL_timm,'mobilenetv3_large_100_miil_in21k'),pretrained_options=[True, False], info_string='mobilenetv3_large_100_miil from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    # 'vit_base_patch16_224_miil': EncoderTuple(partial(prepare_MIIL_timm,'mobilenetv3_large_100_miil_in21k'),pretrained_options=[True, False], info_string='vit_base_patch16_224_miil from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    # 'mixer_b16_224_miil': EncoderTuple(partial(prepare_MIIL_timm,'mobilenetv3_large_100_miil_in21k'),pretrained_options=[True, False], info_string='mixer_b16_224_miil from https://github.com/Alibaba-MIIL/ImageNet21K can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    # # vissl SimCLR    
    # 'vissl_simclr_RN50': EncoderTuple(partial(prepare_vissl,'vissl_simclr_RN50', 'https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch', 'resnext50.yaml'),pretrained_options=[True], info_string='vissl_simclr from VISSL can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    # 'vissl_simclr_RN50_w2': EncoderTuple(partial(prepare_vissl,'vissl_simclr_RN50_w2', 'https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w2_1000ep_simclr_8node_resnet_16_07_20.e1e3bbf0/model_final_checkpoint_phase999.torch', 'resnext50.yaml'),pretrained_options=[True], info_string='vissl_simclr from VISSL can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),
    # 'vissl_simclr_RN50_w4': EncoderTuple(partial(prepare_vissl,'vissl_simclr_RN50_w4', 'https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w4_1000ep_bs32_16node_simclr_8node_resnet_28_07_20.9e20b0ae/model_final_checkpoint_phase999.torch', 'resnext50.yaml'),pretrained_options=[True], info_string='vissl_simclr from VISSL can be pretrained with ImageNet1K', pretrain_dataset='ImageNet1K'),

}

if __name__=='__main__': 
    outF = open("models_info.txt", "w")
    outF.write("Currently available models:\n")
    for ds_encoder in encoders:
        name = ds_encoder
        encoder_tupe = encoders[name]
        outF.write(f"{name}: {encoder_tupe.info_string}")
        outF.write("\n")
    outF.write("="*100)
    outF.write("\n")
    for ds_encoder in encoders:
        name = ds_encoder
        print(name)
        encoder_tupe = encoders[name]
        encoder_partial = encoder_tupe.partial_encoder 
        pretrained_options = encoder_tupe.pretrained_options
        info_string = encoder_tupe.info_string

        feature_extractor = encoder_tupe.partial_encoder(pretrained=False)

        outF.write(f"{name}: {info_string}\n")
        outF.write(f"{feature_extractor.encoder} \n")
        outF.write("Parameter count table: \n")
        outF.write(f"{parameter_count_table(feature_extractor.encoder)}\n")
        outF.write("="*100)
        outF.write("\n")
    outF.close()
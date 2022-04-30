### This repo contains code used for the paper "Foundational Models for Continual Learning: An Empirical Study of Latent Replay".
 
Setting up things:  clone the repo, create a vitual envoronment (e.g. using conda), install the **requirements.txt**.

---
## Dataset encoding

The main file for dataset encoding is the **dataset_encoder.py**.

Usage examples:
    
To encode CIFAR100 dataset with the RN50_clip encoder into an **.hdf5** file locader under \[data_path\]/EncodedDatasets/, use the command:
```
python dataset_encoder.py --pretrained_encoder 1 --regime latent_ER --dataset_name CIFAR100 --dataset_encoder_name RN50_clip
```
          
See **example.py** for how to iterate over the encoded datasets.
 
----
The list of available encoders can be found under [Models/encoders.py](https://github.com/oleksost/latent_CL/blob/master/models/encoders.py). A list of currently available datasets can be found in [Data/datasets.py](https://github.com/oleksost/latent_CL/blob/master/Data/datasets.py)
 
To add new encoder (feature extractor):
1. Add an [EncoderTuple]() with information about the new feature extractor to the list of encoders [here](). 
2. The first argument 'partial_encoder' should be a pointer to a (partialy initialized) function that prepares the new feature encoder (see e.g. an example of [prepare_dino]() function). This function should return an instance of [PreparedModel]() class.
----
    
## Training a classifier with latent ER

The main file for clasifier training and end2end fine-tuning is the **main.py**.
          
Usage example:

To train an MLP classifier on CIFAR100 dataset encoded with ViT-B/16:
```
python main.py --weight_decay 0 --lr_anneal 1 --debug 0 --resolution 100 --optimizer 'adam' --lr $lr --valid_fraction 0.1 --reinit_between_tasks 0 --data_path 'Datasets' --use_predefined_orderings 1 --n_task_order_permutations 5 --er_buffer_type 'balanced' --weights_path 'Weights' --regime latent_ER --epochs 100 --dataset_name 'CIFAR100' --n_tasks 5 --dataset_encoder_name 'ViT-B/16' --cls_hidden_size 1024 --freeze_encoder 1 --cls_n_hidden 1 --classifier_type 'fc' --er_size_per_class 20 --pretrained_encoder 1 --wandb_project 'test' --fraction_buffer_samples_valid 0--task_level_cv 1 --keep_best_params_after_first_task 1
```
To train an MLP classifier on Cars196 dataset encoded with ViT-B/16_clip:

```
python main.py --weight_decay 0 --lr_anneal 1 --debug 0 --resolution 100 --optimizer 'adam' --lr $lr --valid_fraction 0.1 --reinit_between_tasks 0 --data_path 'Datasets' --use_predefined_orderings 1 --n_task_order_permutations 5 --er_buffer_type 'balanced' --weights_path 'Weights' --regime latent_ER --epochs 100 --dataset_name 'Car196' --n_tasks 5 --dataset_encoder_name 'ViT-B/16_clip' --cls_hidden_size 1024 --freeze_encoder 1 --cls_n_hidden 1 --classifier_type 'fc' --er_size_per_class 20 --pretrained_encoder 1 --wandb_project 'test' --fraction_buffer_samples_valid 0--task_level_cv 1 --keep_best_params_after_first_task 1
```

To train an NMC classifier on Cars196 dataset encoded with ViT-B/16_clip:
```
python main.py --weight_decay 0 --lr_anneal 1 --debug 0 --resolution 100 --optimizer 'adam' --lr $lr --valid_fraction 0.1 --reinit_between_tasks 0 --data_path 'Datasets' --use_predefined_orderings 1 --n_task_order_permutations 5 --er_buffer_type 'balanced' --weights_path 'Weights' --regime latent_ER --epochs 100 --dataset_name 'Car196' --n_tasks 5 --dataset_encoder_name 'ViT-B/16_clip' --cls_hidden_size 1024 --freeze_encoder 1 --cls_n_hidden 1 --classifier_type 'nmc' --er_size_per_class 20 --pretrained_encoder 1 --wandb_project 'test' --fraction_buffer_samples_valid 0--task_level_cv 1 --keep_best_params_after_first_task 1
```

 
To train an MLP classifier on Cars196 dataset encoded with a concatenation of the \[ViT-B/16, ViT-B/16_clip,RN50x16_clip\] models:
```
python main.py --weight_decay 0 --lr_anneal 1 --debug 0 --resolution 100 --optimizer 'adam' --lr $lr --valid_fraction 0.1 --reinit_between_tasks 0 --data_path 'Datasets' --use_predefined_orderings 1 --n_task_order_permutations 5 --er_buffer_type 'balanced' --weights_path 'Weights' --regime latent_ER --epochs 100 --dataset_name 'Car196' --n_tasks 5 --concat_dataset_encoders "ViT-B/16" "ViT-B/16_clip" "RN50x16_clip" --cls_hidden_size 1024 --freeze_encoder 1 --cls_n_hidden 1 --classifier_type 'fc' --er_size_per_class 20 --pretrained_encoder 1 --wandb_project 'test' --fraction_buffer_samples_valid 0--task_level_cv 1 --keep_best_params_after_first_task 1
```
 
Optionally pass a list of learning rates and/or weight decays that will be checked in the hyperparameter tuning phase like *--lrs_cv 0.1 0.001 --weight_decays_cv 0 0.0001 --lr_anneals 0 1*.
 
--- 
## End2end fine-tuning
Usage example:          
To end2end fine-tune ViT-B/16 model on Cars196 dataset:

```     
python main.py --resolution 100 --n_task_order_permutations 1 --use_predefined_orderings 1 --n_epochs_task_level_cv 5 --task_level_cv 1 --keep_best_params_after_first_task 1 --optimizer adam --valid_fraction 0.1 --wandb_notes 'end2end' --lr_anneal 1 --lr 2e-05  --weight_decay 0 --momentum 0.9 --batch_size 16 --data_path 'Datasets' --er_buffer_type balanced --weights_path 'Weights' --regime 'sample_ER' --epochs 8 --dataset_name 'Car196' --encoder_name ViT-B/16 --n_tasks 5 --dataset_encoder_name None --cls_hidden_size 1024 --freeze_encoder 0 --cls_n_hidden 1 --classifier_type 'fc' --er_size_per_class 30 --pretrained_encoder 1 --wandb_project 'test'
```

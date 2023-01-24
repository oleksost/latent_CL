### This code acompanies the paper "Continual Learning with Foundation Models: An Empirical Study of Latent Replay" [arxiv](https://arxiv.org/abs/2205.00329).
      
Selected for the [Oral](https://www.youtube.com/watch?v=jWMLZuaRu0E&ab_channel=ConferenceonLifelongLearningAgents%28CoLLAs%29) presentation at [CoLLAs 2022](https://lifelong-ml.cc/Conferences/2022) and published in [PMLR](https://proceedings.mlr.press/v199/ostapenko22a.html).
  
### Abstract:
Rapid development of large-scale pre-training has resulted in foundation models that can act as effective feature extractors on a variety of downstream tasks and domains. Motivated by this, we study the efficacy of pre-trained vision models as a foundation for downstream continual learning (CL) scenarios. Our goal is twofold. First, we want to understand the compute-accuracy trade-off between CL in the raw-data space and in the latent space of pre-trained encoders. Second, we investigate how the characteristics of the encoder, the pre-training algorithm and data, as well as of the resulting latent space affect CL performance. For this, we compare the efficacy of various pre-trained models in large-scale benchmarking scenarios with a vanilla replay setting applied in the latent and in the raw-data space. Notably, this study shows how transfer, forgetting, task similarity and learning are dependent on the input data characteristics and not necessarily on the CL algorithms. First, we show that under some circumstances reasonable CL performance can readily be achieved with a non-parametric classifier at negligible compute. We then show how models pre-trained on broader data result in better performance for various replay sizes. We explain this with representational similarity and transfer properties of these representations. Finally, we show the effectiveness of self-supervised pre-training for downstream domains that are out-of-distribution as compared to the pre-training domain. We point out and validate several research directions that can further increase the efficacy of latent CL including representation ensembling. The diverse set of datasets used in this study can serve as a compute-efficient playground for further CL research.

---
To set things up clone the repo, create a vitual envoronment (e.g. using conda and python 3.7), install the **requirements.txt**.

---
## Dataset encoding.

The main file for dataset encoding is the [**dataset_encoder.py**](https://github.com/oleksost/latent_CL/blob/dcba9e7424ff5f6452ddd905904e521c4f29f11d/Models/encoders.py#L450).

Usage examples:
    
To encode CIFAR100 dataset with the RN50_clip encoder into an **.hdf5** file locader under \[data_path\]/EncodedDatasets/, use the command:
```
python dataset_encoder.py --pretrained_encoder 1 --regime latent_ER --dataset_name CIFAR100 --dataset_encoder_name RN50_clip
```
          
See **example.py** for how to iterate over the encoded datasets.
 
----
The list of available encoders can be found under [Models/encoders.py](https://github.com/oleksost/latent_CL/blob/main/Models/encoders.py). A list of currently available datasets can be found in [Data/datasets.py](https://github.com/oleksost/latent_CL/blob/main/Data/datasets.py)
 
To add new encoder (i.e. feature extractor):  
1. Add an [EncoderTuple]() with information about the new feature extractor to the list of encoders [here](https://github.com/oleksost/latent_CL/blob/dcba9e7424ff5f6452ddd905904e521c4f29f11d/Models/encoders.py#L450). 
2. The first argument 'partial_encoder' should be a pointer to a (partialy initialized) function that prepares the new feature encoder (see e.g. an example of [prepare_dino](https://github.com/oleksost/latent_CL/blob/dcba9e7424ff5f6452ddd905904e521c4f29f11d/Models/encoders.py#L202) function). This function should return an instance of [PreparedModel](https://github.com/oleksost/latent_CL/blob/dcba9e7424ff5f6452ddd905904e521c4f29f11d/Models/encoders.py#L36) class.
----
     
## Training a classifier with latent ER

In the latent ER regime we use a pretrained feature encoder as a data pre-trpcessing step (the encoder is never trained). We perform continual learning with replay on the data encoded by one (or potentially a mixture) of the pretrained encoder(s).

The main file for clasifier training and end2end fine-tuning is the **main.py**.
          
Usage examples:

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
In the end2end finetuning regiume we train botht eh classifier and the (pretrained) feature encoder.

Usage example:          
To perform end2end fine-tuning of the ViT-B/16 model on Cars196 dataset run the following command:

```     
python main.py --resolution 100 --n_task_order_permutations 1 --use_predefined_orderings 1 --n_epochs_task_level_cv 5 --task_level_cv 1 --keep_best_params_after_first_task 1 --optimizer adam --valid_fraction 0.1 --wandb_notes 'end2end' --lr_anneal 1 --lr 2e-05  --weight_decay 0 --momentum 0.9 --batch_size 16 --data_path 'Datasets' --er_buffer_type balanced --weights_path 'Weights' --regime 'sample_ER' --epochs 8 --dataset_name 'Car196' --encoder_name ViT-B/16 --n_tasks 5 --dataset_encoder_name None --cls_hidden_size 1024 --freeze_encoder 0 --cls_n_hidden 1 --classifier_type 'fc' --er_size_per_class 30 --pretrained_encoder 1 --wandb_project 'test'
```

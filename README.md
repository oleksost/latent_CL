This repo contains code used for the paper "Foundational Models for Continual Learning: An Emperical Study of Latent Replay".

Setting up things:  clone the repo, create a vitual envoronment (e.g. using conda), install the **requirements.txt**. Currently, this repo only contains the dataset encoding routines.
 
The main file is the **dataset_encoder.py**.

Usage examples:
    
To encode CIFAR100 dataset with the RN50_clip encoder into an **.hdf5** file locader under \[data_path\]/EncodedDatasets/ run:
```
python dataset_encoder.py --pretrained_encoder 1 --regime latent_ER --dataset_name CIFAR100 --dataset_encoder_name RN50_clip
```

See **example.py** for how to iterate through the encoded datasets.

----
The list of available encoders can be found under [Models/encoders.py](https://github.com/oleksost/latent_CL/blob/master/models/encoders.py). Find a list of currently available datasets in [Data/datasets.py](https://github.com/oleksost/latent_CL/blob/master/Data/datasets.py)

To add new encoder (feature extractor):
1. Add an [EncoderTuple]() with information about the ned feature extractor to list of encoders [here](). 
2. The first argument 'partial_encoder' should be a pointer to a (partialy initialized) function that prepares the new feature encoder (see e.g. an example of [prepare_dino]() function). This function should return a [PreparedModel]() named tuple.
----

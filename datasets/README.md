# Data Preparing

1. Access to the synapse multi-organ dataset:
   1. Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
   2.  You can also send an Email directly to jienengchen01 AT gmail.com to request the preprocessed data for reproduction.
2. The directory structure of the whole project is as follows:

```bash

├── SP+E Model
│   ├──datasets
│   │       └── dataset_vfss.py
│   ├──model
│   │       └── vit_checkpoint
│   │           └── imagenet21k
│   │                └──  R50+ViT-B_16.npz
│   ├──networks
│   │       └── vit_seg_configs.py
│   │       └── vit_seg_modeling_resnet_skip.py
│   │       └── vit_seg_modeling.py
│   ├──train.py
│   ├──test.py
│   ├──trainer.py
│   └──util.py
│   
└── data
    └──train
        ├── img
        │   └── *.jpeg
        └── mask
            └── *.npy
    └──valid
        ├── img
        │   └── *.jpeg
        └── mask
            └── *.npy
    └──test
        ├── img
        │   └── *.jpeg
        └── mask
            └── *.npy



# GoogLeNet-Inception-V1-Implementation

## Architecture

<img src="images/architecture.png" alt="GoogLeNet Architecture" style="width:100%;">

"Going Deeper with Convolutions" by Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke and Andrew Rabinovich.

Paper: https://arxiv.org/pdf/1409.4842.pdf

## Inception Module

<img src="images/inception-module.png" alt="Inception Module" style="width:100%;">

<img src="images/network.png" alt="GoogLeNet Network" style="width:100%;">

## Another day, same old story

GPU POOR !!!

Didn't train cause I don't have a powerful GPU. But the architecture is there for playing.

## Info

Run script below to checkout the model informations

```sh
python info.py
```

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GoogLeNet                                [1, 1000]                 6,379,728
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,472
├─MaxPool2d: 1-2                         [1, 64, 56, 56]           --
├─Conv2d: 1-3                            [1, 64, 56, 56]           4,160
├─Conv2d: 1-4                            [1, 192, 56, 56]          110,784
├─MaxPool2d: 1-5                         [1, 192, 28, 28]          --
├─InceptionModule: 1-6                   [1, 256, 28, 28]          --
│    └─Conv: 2-1                         [1, 64, 28, 28]           --
│    │    └─Conv2d: 3-1                  [1, 64, 28, 28]           12,352
│    └─Sequential: 2-2                   [1, 128, 28, 28]          --
│    │    └─Conv: 3-2                    [1, 96, 28, 28]           18,528
│    │    └─Conv: 3-3                    [1, 128, 28, 28]          110,720
│    └─Sequential: 2-3                   [1, 32, 28, 28]           --
│    │    └─Conv: 3-4                    [1, 16, 28, 28]           3,088
│    │    └─Conv: 3-5                    [1, 32, 28, 28]           4,640
│    └─Sequential: 2-4                   [1, 32, 28, 28]           --
│    │    └─MaxPool2d: 3-6               [1, 192, 28, 28]          --
│    │    └─Conv: 3-7                    [1, 32, 28, 28]           6,176
├─InceptionModule: 1-7                   [1, 480, 28, 28]          --
│    └─Conv: 2-5                         [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-8                  [1, 128, 28, 28]          32,896
│    └─Sequential: 2-6                   [1, 192, 28, 28]          --
│    │    └─Conv: 3-9                    [1, 128, 28, 28]          32,896
│    │    └─Conv: 3-10                   [1, 192, 28, 28]          221,376
│    └─Sequential: 2-7                   [1, 96, 28, 28]           --
│    │    └─Conv: 3-11                   [1, 32, 28, 28]           8,224
│    │    └─Conv: 3-12                   [1, 96, 28, 28]           27,744
│    └─Sequential: 2-8                   [1, 64, 28, 28]           --
│    │    └─MaxPool2d: 3-13              [1, 256, 28, 28]          --
│    │    └─Conv: 3-14                   [1, 64, 28, 28]           16,448
├─MaxPool2d: 1-8                         [1, 480, 14, 14]          --
├─InceptionModule: 1-9                   [1, 512, 14, 14]          --
│    └─Conv: 2-9                         [1, 192, 14, 14]          --
│    │    └─Conv2d: 3-15                 [1, 192, 14, 14]          92,352
│    └─Sequential: 2-10                  [1, 208, 14, 14]          --
│    │    └─Conv: 3-16                   [1, 96, 14, 14]           46,176
│    │    └─Conv: 3-17                   [1, 208, 14, 14]          179,920
│    └─Sequential: 2-11                  [1, 48, 14, 14]           --
│    │    └─Conv: 3-18                   [1, 16, 14, 14]           7,696
│    │    └─Conv: 3-19                   [1, 48, 14, 14]           6,960
│    └─Sequential: 2-12                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-20              [1, 480, 14, 14]          --
│    │    └─Conv: 3-21                   [1, 64, 14, 14]           30,784
├─InceptionModule: 1-10                  [1, 512, 14, 14]          --
│    └─Conv: 2-13                        [1, 160, 14, 14]          --
│    │    └─Conv2d: 3-22                 [1, 160, 14, 14]          82,080
│    └─Sequential: 2-14                  [1, 224, 14, 14]          --
│    │    └─Conv: 3-23                   [1, 112, 14, 14]          57,456
│    │    └─Conv: 3-24                   [1, 224, 14, 14]          226,016
│    └─Sequential: 2-15                  [1, 64, 14, 14]           --
│    │    └─Conv: 3-25                   [1, 24, 14, 14]           12,312
│    │    └─Conv: 3-26                   [1, 64, 14, 14]           13,888
│    └─Sequential: 2-16                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-27              [1, 512, 14, 14]          --
│    │    └─Conv: 3-28                   [1, 64, 14, 14]           32,832
├─InceptionModule: 1-11                  [1, 512, 14, 14]          --
│    └─Conv: 2-17                        [1, 128, 14, 14]          --
│    │    └─Conv2d: 3-29                 [1, 128, 14, 14]          65,664
│    └─Sequential: 2-18                  [1, 256, 14, 14]          --
│    │    └─Conv: 3-30                   [1, 128, 14, 14]          65,664
│    │    └─Conv: 3-31                   [1, 256, 14, 14]          295,168
│    └─Sequential: 2-19                  [1, 64, 14, 14]           --
│    │    └─Conv: 3-32                   [1, 24, 14, 14]           12,312
│    │    └─Conv: 3-33                   [1, 64, 14, 14]           13,888
│    └─Sequential: 2-20                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-34              [1, 512, 14, 14]          --
│    │    └─Conv: 3-35                   [1, 64, 14, 14]           32,832
├─InceptionModule: 1-12                  [1, 528, 14, 14]          --
│    └─Conv: 2-21                        [1, 112, 14, 14]          --
│    │    └─Conv2d: 3-36                 [1, 112, 14, 14]          57,456
│    └─Sequential: 2-22                  [1, 288, 14, 14]          --
│    │    └─Conv: 3-37                   [1, 144, 14, 14]          73,872
│    │    └─Conv: 3-38                   [1, 288, 14, 14]          373,536
│    └─Sequential: 2-23                  [1, 64, 14, 14]           --
│    │    └─Conv: 3-39                   [1, 32, 14, 14]           16,416
│    │    └─Conv: 3-40                   [1, 64, 14, 14]           18,496
│    └─Sequential: 2-24                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-41              [1, 512, 14, 14]          --
│    │    └─Conv: 3-42                   [1, 64, 14, 14]           32,832
├─InceptionModule: 1-13                  [1, 832, 14, 14]          --
│    └─Conv: 2-25                        [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-43                 [1, 256, 14, 14]          135,424
│    └─Sequential: 2-26                  [1, 320, 14, 14]          --
│    │    └─Conv: 3-44                   [1, 160, 14, 14]          84,640
│    │    └─Conv: 3-45                   [1, 320, 14, 14]          461,120
│    └─Sequential: 2-27                  [1, 128, 14, 14]          --
│    │    └─Conv: 3-46                   [1, 32, 14, 14]           16,928
│    │    └─Conv: 3-47                   [1, 128, 14, 14]          36,992
│    └─Sequential: 2-28                  [1, 128, 14, 14]          --
│    │    └─MaxPool2d: 3-48              [1, 528, 14, 14]          --
│    │    └─Conv: 3-49                   [1, 128, 14, 14]          67,712
├─MaxPool2d: 1-14                        [1, 832, 7, 7]            --
├─InceptionModule: 1-15                  [1, 832, 7, 7]            --
│    └─Conv: 2-29                        [1, 256, 7, 7]            --
│    │    └─Conv2d: 3-50                 [1, 256, 7, 7]            213,248
│    └─Sequential: 2-30                  [1, 320, 7, 7]            --
│    │    └─Conv: 3-51                   [1, 160, 7, 7]            133,280
│    │    └─Conv: 3-52                   [1, 320, 7, 7]            461,120
│    └─Sequential: 2-31                  [1, 128, 7, 7]            --
│    │    └─Conv: 3-53                   [1, 32, 7, 7]             26,656
│    │    └─Conv: 3-54                   [1, 128, 7, 7]            36,992
│    └─Sequential: 2-32                  [1, 128, 7, 7]            --
│    │    └─MaxPool2d: 3-55              [1, 832, 7, 7]            --
│    │    └─Conv: 3-56                   [1, 128, 7, 7]            106,624
├─InceptionModule: 1-16                  [1, 1024, 7, 7]           --
│    └─Conv: 2-33                        [1, 384, 7, 7]            --
│    │    └─Conv2d: 3-57                 [1, 384, 7, 7]            319,872
│    └─Sequential: 2-34                  [1, 384, 7, 7]            --
│    │    └─Conv: 3-58                   [1, 192, 7, 7]            159,936
│    │    └─Conv: 3-59                   [1, 384, 7, 7]            663,936
│    └─Sequential: 2-35                  [1, 128, 7, 7]            --
│    │    └─Conv: 3-60                   [1, 48, 7, 7]             39,984
│    │    └─Conv: 3-61                   [1, 128, 7, 7]            55,424
│    └─Sequential: 2-36                  [1, 128, 7, 7]            --
│    │    └─MaxPool2d: 3-62              [1, 832, 7, 7]            --
│    │    └─Conv: 3-63                   [1, 128, 7, 7]            106,624
├─AvgPool2d: 1-17                        [1, 1024, 1, 1]           --
├─Dropout: 1-18                          [1, 1024, 1, 1]           --
├─Linear: 1-19                           [1, 1000]                 1,025,000
==========================================================================================
Total params: 12,997,352
Trainable params: 12,997,352
Non-trainable params: 0
Total mult-adds (G): 1.50
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 25.82
Params size (MB): 26.47
Estimated Total Size (MB): 52.89
==========================================================================================
```

## Usage

Before running the script, place your data directory location for both train and test data in `root_dir="{DIR}"` here at [dataloader.py](./dataloader/dataloader.py)

```sh
python train.py --epochs 100
```
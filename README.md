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

<img src="images/info1.png" alt="GoogLeNet Network" style="width:100%;">
<img src="images/info2.png" alt="GoogLeNet Network" style="width:100%;">
<img src="images/info3.png" alt="GoogLeNet Network" style="width:100%;">

## Usage

Before running the script, place your data directory location for both train and test data in `root_dir="{DIR}"` here at [dataloader.py](./dataloader/dataloader.py)

```sh
python train.py --epochs 100
```
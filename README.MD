# Jointly-learned Exit and Inference for Dynamic Neural Networks

This repository holds companion code for our paper [Jointly-learned Exit and Inference for Dynamic Neural Networks](https://arxiv.org/abs/2310.09163). We explore ways of training an efficient dynamic neural network by augmenting a frozen off-the-shelf neural network as backbone.


## Supported datasets
Out of the box this codebase supports CIFAR10, CIFAR100, SVHN and CIFAR100-LT.

## Supported models
Out of the box we support T2T-ViT-7 and T2T-ViT-14. You can add other models using [timm](https://github.com/huggingface/pytorch-image-models).

## Running the code
First, install all requirements `pip install -r requirements.txt` (it is preferable if you create a virtual environment using conda or venv and install those requirements in that environment)
You can then choose to run JEI-DNN directly on some of the already supported datasets or transfer learn an Imagenet-pretrained version onto a new dataset.
### Transfer learning
1. Download the weights for the 7-layer vision transformer T2T-ViT-7 trained on Imagenet:  from https://github.com/yitu-opensource/T2T-ViT and store them locally. 
2. To run the following, making sure you update the `--weights-path` parameter below:
```
python transfer_learning.py --lr 0.05 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --weights-path model_weights/71.7_T2T_ViT_7.pth.tar
```

### Run JEI-DNN on supported datasets.
1. Download the model checkpoints from this [google drive](https://drive.google.com/drive/folders/1kyAVHtTKNx-h_pgpm_9cqktF8Ce6ZiAg?usp=sharing)  and place each in the appropriate checkpoint folder (by dataset and architecture).
2. Make sure the path match the path in train_dynn.py for the dataset.
3. You can now train JEI-DNN for dynamic inference. You can specify the dataset, the architecture, number of epochs and the `ce_ic_tradeoff`. Higher values of `ce_ic_tradeoff` mean the model is trained to exit earlier, at the cost of losing accuracy.
```
python train_dynn.py --ce_ic_tradeoff 0.15 --dataset cifar10 --arch t2t_vit_7 --num_epoch 15;
```

## Start mlflow ui
We used [mlflow](https://www.mlflow.org/docs/latest/index.html) to monitor our running scripts.
1. Install mlflow (it is listed in our requirements.txt)
2. To start the ui `mlflow ui` (from the root of the project)

## Attribution
The starting code was taken from the original repository for [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT) since we used T2T-ViT-7/14 models as backbone for our dynamic neural network.

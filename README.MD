# Jointly-learned Exit and Inference for Dynamic Neural Networks

This repository holds companion code for our paper "Jointly-learned Exit and Inference for Dynamic Neural Networks". We explore ways of training an efficient dynamic neural network by augmenting a frozen off-the-shelf neural network as backbone.

## Attribution
The starting code was taken from the original repository for [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT) since we used various T2T-ViT models as backbone for our dynamic neural network. Some code was borrowed from [Boosted-Dynamic-Networks](https://github.com/SHI-Labs/Boosted-Dynamic-Networks) and [L2W-DEN](https://github.com/LeapLabTHU/L2W-DEN) which we used as baselines.

## Supported datasets
Out of the box this codebase supports CIFAR10, CIFAR100, SVHN and CIFAR100-LT.

## Running the code
To run the code:
1. Install all requirements `pip install -r requirements.txt`
2. Download the weights for the 7-layer vision transformer T2T-ViT-7 trained on Imagenet:  from https://github.com/yitu-opensource/T2T-ViT and store them locally.
3. To transfer learn T2T-ViT-7 to CIFAR-10, you need to run `transfer_learning.py` with the right parameters including `--weights-path` which points to the directory where you downloaded the weights, starting from the root of the project.
```
python transfer_learning.py --lr 0.05 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --weights-path model_weights/71.7_T2T_ViT_7.pth.tar
```
4. Once your model was transfer-learned, you can train it for dynamic inference. You can specify the dataset, the architecture, number of epochs and the `ce_ic_tradeoff`. Higher values of `ce_ic_tradeoff` mean the model is trained to exit earlier, at the cost of losing accuracy.
```
python train_dynn.py --ce_ic_tradeoff 0.15 --dataset cifar10 --arch t2t_vit_7 --num_epoch 15;
```

## Start mlflow ui
We used [mlflow](https://www.mlflow.org/docs/latest/index.html) to monitor our running scripts.
1. Install mlflow (it is listed in our requirements.txt)
2. To start the ui `mlflow ui` (from the root of the project)
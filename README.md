# DyNN
DyNN stands for dynamic neural network. Dynamic neural networks reduce the average cost of inference by adapting the complexity of the network to the input, using the most costly path only for the most complex inputs.

## Setup and running
Download the weights for the 7-layer vision transformer T2T-ViT-7 trained on Imagenet:  from https://github.com/yitu-opensource/T2T-ViT and store them locally.

To transfer learn T2T-ViT-7 to CIFAR-10, you need to run `transfer_learning.py` with the right parameters including `--transfer-model` which points to the directory where you downloaded the weights.
```
python transfer_learing --lr 0.05 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --transfer-model model_weights/71.7_T2T_ViT_7.pth.tar
```

## Run mlflow

``
mlflow server --backend-store-uri ./mlruns --port 5002
''

# Use the plotting notebook

1- Generate the plots in the mlflow web tool
2- Download the csv file
3- Replace the name in the generate_plot_mlflow.ipynb notebook

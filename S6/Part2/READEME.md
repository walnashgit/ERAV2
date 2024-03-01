# MNIST classification

## This folder contains a neural network that works on MNIST data for classification of numbers from 0 - 9.

## Introduction

The model and the supporting code in 99_41_Avi_EVA4_Session_6.ipynb file trains and validate on MNIST data. The target of this model is to achieve 99.4% validation accuracy. 

## Model description

The model used in this file contains of **6 layers**. It utilises Batch normalisation, Dropout and a Fully connected layer with Global Average Pooling.
The layers are as described:

### Layer 1
The first part in first layer is the convolution of the input data. 

    Input channel: 1
    Output channel: 32
    Kernel size: 3
    RF: 3
    param: 288

The second part is batch normalisation of 32 features (no. of output channel in the previous operation)

    param: 64
The third part is dropout with 5% probability. _This model uses dropout with 5% probability_.

### Layer 2
The first part in second layer is the convolution of the data from previous layer

    Input channel: 32
    Output channel: 32
    Kernel size: 3
    RF: 5
    param: 9216

The second part is batch normalisation of 32 features (no. of output channel in the previous operation)

    param: 64
The third part is maxpool.

    RF: 6

The fourth part is dropout with 5% probability.

### Layer 3
The first part in third layer is the convolution of the data from previous layer

    Input channel: 32
    Output channel: 16
    Kernel size: 3
    RF: 10
    param: 4608

The second part is batch normalisation of 16 features (no. of output channel in the previous operation)

    param: 32
The third part is maxpool.

    RF: 12

### Layer 4
In the fourth layer we only have convolution of the data from previous layer

    Input channel: 16
    Output channel: 16
    Kernel size: 3
    RF: 20
    param: 2304

### Layer 5
In the fifth layer we only have convolution of the data from previous layer

    Input channel: 16
    Output channel: 16
    Kernel size: 3
    RF: 28
    param: 2304

### Layer 6
Layer 6 is a combination of Global Average Pooling and a fully connected layer.
    
    param: 170 (from FC layer)
    
## Other values used in this model:

    Batch size for training and validation: 128
    Learning Rate (LR): 0.01
    momentum: 0.9
    Epoch used: 19








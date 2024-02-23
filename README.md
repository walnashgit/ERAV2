# ERAV2

## This repository contains the codes for a neural network that works on MNIST data for classification of numbers from 0 - 9. 

## Introduction

The repo comprises of 3 files. 
1. model.py
2. utils.py
3. S5.ipynb

S5.ipynb file contains the code to train the model in model.py using the functions from utils.py

## model.py
This file has a Network class defined that implements a convolutional neural network (CNN) architecture. 

1. **Convolutional Layers**:
  - conv1: Applies a convolutional operation with 32 filters and a kernel size of 3x3.
  - conv2: Applies a convolutional operation with 64 filters and a kernel size of 3x3.
  - conv3: Applies a convolutional operation with 128 filters and a kernel size of 3x3.
  - conv4: Applies a convolutional operation with 256 filters and a kernel size of 3x3.

2. **Fully Connected Layers**:
  - fc1: Linear layer with 4096 input features and 50 output features.
  - fc2: Linear layer with 50 input features and 10 output features (final output).

**Forward Pass**

The `forward` method of the `Network` class defines the forward pass of the network:

    Applies ReLU activation after each convolutional layer.
    Applies max pooling after certain convolutional layers.
    Flattens the output tensor before passing it to fully connected layers.
    Applies ReLU activation to the output of the first fully connected layer.
    Computes the final output logits using the second fully connected layer.
    Applies log softmax activation to the output logits along dimension 1 (across classes).


## utils.py
This file contains all the helper function needed to train and evaluate the CNN present in model.py.

### Functions in utils.py
- **getTrainingTransforms()**: Returns transformations for preprocessing training data.
- **getTestTransforms()**: Returns transformations for preprocessing test data.
- **getTrainingData(train_transforms)**: Downloads and returns the MNIST training dataset with specified transformations.
- **getTestData(test_transforms)**: Downloads and returns the MNIST test dataset with specified transformations.
- **getDataLoader(data, batch_size)**: Returns a DataLoader created with the provided data and batch size.
- **plotTrainDataSample(train_loader, sample_size)**: Plots a sample of training data for visualization.
- **plotTrainingTestLossAndAccuracy()**: Plots the graph for training and test loss and accuracy.
- **printModelSummary(model)**: Prints the summary of the model architecture using torchsummary.
- **train(model, device, train_loader, optimizer, criterion)**: Trains the model with the provided training data.
- **test(model, device, test_loader, criterion)**: Tests the trained model against the provided test data.
- **GetCorrectPredCount(pPrediction, pLabels)**: Returns the count of correct predictions.
- **getAvailableDevice()**: Returns the available device (CPU or GPU) for computation.


## S5.ipynb
This is a colab notebook file that follow a step by step process to train and test the nueral netwrok in model.py file using the various functions present in utils.py file. It follows the following steps:

- Mounting Google Drive to access required files (`gdrive` and `drive` imports).
- Importing the necessary modules and functions from the ERA package (`model.py` and `utils.py`).
- Setting up training and test data using `getTrainingData()` and `getTestData()` functions from `utils.py`.
- Creating data loaders for training and test data using `getDataLoader()` function.
- Visualizing a sample of training data using `plotTrainDataSample()` function.
- Setting up the model architecture using the `Network` class from `model.py` and printing its summary with `printModelSummary()` function.
- Defining the optimizer, scheduler, and loss criterion for training the model.
- Training and testing the model for a specified number of epochs.
- Plotting training and test loss and accuracy using `plotTrainingTestLossAndAccuracy()` function.


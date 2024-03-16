# How to build a DNN - Model-2

### This is the second part in the building of a DNN model. In this part we'll improve our previous model by adding GAP, BN and regularization. Focus will be to reduce overfitting.

## Introduction
This folder contains 2 files:


1.   [S7_model_2.ipynb](https://github.com/walnashgit/ERAV2/blob/main/S7/model2/S7_model_2.ipynb)
2.   [model_2.py](https://github.com/walnashgit/ERAV2/blob/main/S7/model2/model_2.py)

The .ipynb file contains the code for data, transform, training and testing the model.<br>
The model_2.py file contains the model.


## Target:

1.   Reduce overfitting by using diffrent architecture components like Batch Normalization, Dropout(Regularization) and Global Average Pooling.


## Result

1. Total params: 7,580
2. Best training accuracy: 97.97% (20th Epoch)
3. Best test accuracy: 98.93% (20th Epoch)


## Analysis

1. Model is lighter (7.5k Params).
2. No overfitting is there.
3. Test and training accuracy is not very high but can increase if model is pushed further.
4. Increasing the capacity and placing some of the componenets like MaxPool at right location may help.
5. Making training a bit more difficult by adding image augmentation may also improve accuracy.





# How to build a DNN - Model-3

### This is the third part in the building of a DNN model. In this part we'll try to improve the accuracy of our previous model by tweaking capacity and adding image augmentation

## Introduction
This folder contains 2 files:


1.   [S7_model_3.ipynb](https://github.com/walnashgit/ERAV2/blob/main/S7/model3/S7_model_3.ipynb)
2.   [model_3.py](https://github.com/walnashgit/ERAV2/blob/main/S7/model3/model_3.py)

The .ipynb file contains the code for data, transform, training and testing the model.<br>
The model_3.py file contains the model.


## Target:

1.   Improve accuracy by adding or tweaking capacity and adding Image augmentation.


## Result

1. Total params: 7,560
2. Best training accuracy: 98.30%
3. Best test accuracy: 99.30% 


## Analysis

1. Model is not overfitting. 
2. Model accuracy has improved slightly but still lower than our target of 99.4%.
3. Test and training accuracy can increase if model is pushed further.
4. Increasing the capacity may help but we are almost at the limit of 8k param.
5. Tweaking learning rate or using a scheduler for it may help.

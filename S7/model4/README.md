# How to build a DNN - Model-4

### This is the fourth part in the building of a DNN model. In this part we'll try to improve the accuracy of our previous model further by adjusting leraing rate. We'll try to achieve our target of 99.4% in under 15 EPOCHS.

## Introduction
This folder contains 2 files:


1.   [S7_model_4.ipynb](https://github.com/walnashgit/ERAV2/blob/main/S7/model4/S7_model_4.ipynb)
2.   [model_3.py](https://github.com/walnashgit/ERAV2/blob/main/S7/model3/model_3.py)

The .ipynb file contains the code for data, transform, training and testing the model.<br>
The model_3.py file contains the model.(same as model_3)


## Target:

1. Improve accuracy by adjusting learning rate.


## Result (considering only first 15 epochs)

1. Total params: 7,580
2. Best training accuracy: 98.58% (Epoch - 11)
3. Best test accuracy: 99.35% (Epoch - 7)



## Analysis

1. Model accuracy seems to have stablized around 99.35%.
2. Tweaking LR has helped but it's tweaked very naively. A dynamic approach based on data may help better.
3. We could not achieve our target of 99.4%
4. Increasing capacity and some modification to architecture may help improving the accuracy.

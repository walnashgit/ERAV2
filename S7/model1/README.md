# How to build a DNN - Model-1

### This is the first part in the building of a DNN model. This part will cover basic setup, a skeletal model with limited but not very low number of  parameters.

## Introduction
This folder contains 2 files:


1.   [S7_model_1.ipynb](https://github.com/walnashgit/ERAV2/blob/main/S7/model1/S7_model_1.ipynb)
2.   [model_1.py](https://github.com/walnashgit/ERAV2/blob/main/S7/model1/model_1.py)

The .ipynb file contains the code for data, transform, training and testing the model.<br>
The model_1.py file contains the basic model. The n


## Target:

1.   Create the basic setup for data, transform, model, training and test code
2.   Create an initial model focussed on achieving the required receptive field (min 28) and keeping the model light with total params under 15k.


## Result

1. Total params: 10,066
2. Best training accuracy: 99.5%
3. Best test accuracy: 99.12%

## Analysis

1. Model is light (only 10k param).
2. Bit of overfitting is there.
3. Doesnt look like model would improve test accuracy even if run for more epochs as it has mostly stagnated at around 99%. Training accuracy has also stagnated around 99.5%.
4. We can improve the model by adding some architectural components like GAP and with regularization.




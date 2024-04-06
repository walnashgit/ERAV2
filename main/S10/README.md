# Custom ResNet with One Cycle Policy

This code demonstrates a custom resnet model and is trained using one cycle policy. Data trained is CIFAR10.

The custom resnet model in [S10CustomResnetModel.py](https://github.com/walnashgit/ERAV2/blob/main/main/S10/S10ResnetOneCycle.ipynb) follows the below architecture:

1:  PrepLayer - Conv 3x3 (s1, p1) >> BN >> RELU [64k]

2:  Layer1 -

    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    Add(X, R1)

3:  Layer 2 -

    Conv 3x3 [256k]
    MaxPooling2D
    BN
    ReLU

4:  Layer 3 -

    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    Add(X, R2)

5:  MaxPooling with Kernel Size 4

6:  FC Layer 



The code in [S10ResnetOneCycle](https://github.com/walnashgit/ERAV2/blob/main/main/S10/S10ResnetOneCycle.ipynb) handles the data, training and validation of the model. 

## One Cycle Policy

This model is trained using One Cycle Policy where the Learning Rate (LR) is varied during the learning phase of the model. In this code, the model is trained for 24 epochs with a batch size of 512.

The LR is started at LR_max/10 and increased to LR_max in 4 epochs (LR_max will be used in 5th epoch) and then reduced back to LR_max/10 in remaining 19 epochs.

LR_max is found using [fastai](https://github.com/davidtvs/pytorch-lr-finder?tab=readme-ov-file#tweaked-version-from-fastai)

The code for finding LR_max is in [Utils.py](https://github.com/walnashgit/ERAV2/blob/dc9326331bd097a0470655376161b263a4f3dcde/main/Utils.py#L81C9-L81C23)

Target for this model is 90% accuracy.

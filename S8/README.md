# Image classification (CIFAR10 dataset) using Batch, Group and Layer normalization.

## The folder S8 contains 3 sub folder:


1.   Batch Normalization.
2.   Group Normalization.
3.   Layer Normalization.

## Introduction

All the three models in the three approach follow the below structure:

C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11

where c3, c7 and c11 are 1x1 convolution.

The difference between the 3 models is the normalization technique. 

The models aims to classify the images in CIFAR10 dataset. The aim of the models is to achieve 70% accuracy using 50K or less params in under 20 EPOCHS.

## Batch normalization


The model in the S8CIFAR_BN.ipynb is a convolution network that uses Batch Normalization. <br>
**Total Parms: 48096** <br>
**Training accuracy: 70.82%** <br>
**Test accuracy: 74.5%**

### Training and Test Graphs
<img width="915" alt="BatchNormGraphs" src="https://github.com/walnashgit/ERAV2/assets/73463300/1933d519-7a6a-4b6f-9f0f-c18ed06674d0">

### Missclassified images
<img width="566" alt="BatchNormMissClassified" src="https://github.com/walnashgit/ERAV2/assets/73463300/d89944d1-e98f-4c0d-a249-4e43ee9acbd5">


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

## Group normalization

The model in the S8CIFAR_GN.ipynb is a convolution network that uses Group Normalization. <br>
**Total Parms: 48096** <br>
**Training accuracy: 71.01%** <br>
**Test accuracy: 73.08%**

### Training and Test Graphs
<img width="921" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/3fdd1878-36f0-40fb-87a6-1a636490088a">

### Missclassified images
<img width="563" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/13fbee57-2910-4d95-a0cd-f7fcecbe130f">

## Layer normalization

The model in the S8CIFAR_LN.ipynb is a convolution network that uses Layer Normalization. <br>
**Total Parms: 111,568** <br>
**Training accuracy: 52.05%** <br>
**Test accuracy: 59.44%**

### Training and Test Graphs
<img width="918" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/2516fc7d-57a0-422c-9095-c59ce0a46f87">

### Missclassified images
<img width="564" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/5acc03ea-a685-4562-b016-001b71ca95ce">

## Observation


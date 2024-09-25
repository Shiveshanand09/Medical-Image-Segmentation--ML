# Medical Image Segmentation 

## DATASET

### HyperKvasir: A Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy

Gastrointestinal endoscopy is a type of endoscopic procedure that allows physicians to examine the digestive system by inserting a long, flexible lighted instrument called an endoscope either through the rectum or down through the throat.
A polyp is a projecting growth of tissue from a surface in the body, usually a mucous membrane. Bowel polyps are not usually cancerous, although if they're discovered they'll need to be removed, as some will eventually turn into cancer if left untreated.
This model is capable of depicting polyp tissue for Gastrointestinal Endoscopy. 

Download dataset from here: https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip

## Data augmentation

The data for training contains 1000 128*128 images, which are far not enough to feed a deep learning neural network. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.

See data_preprocessing.py and train.py for detail.

## MODEL

![u-net-architecture](https://github.com/GabruAru/Medical-Image-Segmentation/assets/84130891/067ca172-27d6-449b-9ed9-97dd0faee096)


**UNET — Network Architecture**

UNET is a U-shaped encoder-decoder network architecture, which consists of four encoder blocks and four decoder blocks that are connected via a bridge. The encoder network (contracting path) half the spatial dimensions and double the number of filters (feature channels) at each encoder block. Likewise, the decoder network doubles the spatial dimensions and half the number of feature channels.

### Used Architecture

![architecture](https://github.com/GabruAru/Medical-Image-Segmentation/assets/84130891/71e03bc0-3a41-4b1b-a9d9-9ee53c6305b7)


## Training

The model is trained for 100 epochs. Using Binary crossentropy as loss function achiving aroung 90% accuracy. Used Learning rate is 0.001(default for Adam).

Then model is trained further using dice coefficient as loss function achieving accuracy of 97%. And Mean IOU of around 91. Used Learning rate here is 0.0001.


## How to use

1. run data_preprocessing.py to import dataset

2. run model.py to train model
   
3. run evaluation.py for reuslts


# Results 

Got good results using trained model.

![Evaluation](https://github.com/GabruAru/Medical-Image-Segmentation/assets/84130891/79ba60c4-641d-462f-b115-fd44ea082104)
![Evaluation1](https://github.com/GabruAru/Medical-Image-Segmentation/assets/84130891/ffe6bf09-819b-4e47-aa5f-06e5503ffc71)


# Evalutaion metrics

Mean IoU = 0.926379

BinaryAccuracy = 0.983

Precision Score = 0.953

Recall Score = 0.938

F1 Score = 0.945

# Image Segmentation with U-Net

This repository contains a Jupyter Notebook implementation of image segmentation using the U-Net architecture. U-Net is widely used for segmentation tasks, especially in the field of biomedical image segmentation but can be extended to other domains as well.

## Table of Contents
1. [Overview](#overview)
2. [U-Net Architecture](#u-net-architecture)
3. [Requirements](#requirements)
4. [Installation and Setup](#installation-and-setup)
5. [Dataset](#dataset)
6. [Training the Model](#training-the-model)
7. [Evaluating the Model](#evaluating-the-model)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [License](#license)

---

## Overview

Image segmentation is a fundamental problem in computer vision where the goal is to partition an image into meaningful segments or regions, typically corresponding to objects or regions of interest.

This project demonstrates how to build, train, and evaluate a U-Net model for image segmentation. It supports custom datasets and provides tools for preprocessing, training, and evaluating the model with detailed metrics like IoU (Intersection over Union) and Dice coefficient.

## U-Net Architecture

The U-Net architecture consists of two parts:

1. **Encoder (Contracting Path)**: This reduces the spatial dimension of the image by applying convolutional layers followed by max-pooling. It captures context with each layer.
2. **Decoder (Expanding Path)**: This upsamples the features back to the original image size using transpose convolutions and concatenates them with high-resolution features from the encoder. This step recovers spatial information lost during downsampling.
   
The architecture is visualized as follows:

```text
Input Image --> [Encoder] --> Bottleneck --> [Decoder] --> Output Mask


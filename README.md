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

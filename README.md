# Implementing Response Based Knowledge distillation on Two CNN models
This project talks about the application of Knowledge Distillation using temperatue to modify the soft max outputs.

## Table of Contents
- Overview
- Data
- Model Architecture
- Results

## Overview
Response based knowledge Distillation is a type of Knowledge Distillation that uses temperature to modify the softmax outputs, The temperature parameter controls how much informaton from the teacher's output distribution gets transferred to the student . It's called response based because it work's with the model's final responses. 
Deep Neural networks achieve state of the art performance across different computer vision tasks but are too computationally intensive for embedded devices. This project applies Knowledge distillation techniques to compress a large pre-trained CNN model into a smaller one while maintaining acceptable performance. I implemented temperature based response knowledge distillation on a pretrained model using the Pytorch's implementation as a  guide, transferring knowledge from a larger teacher model to a more smaller student model for image classification.

## Data 
- Data description
The dataset used  was a  CIFAR 10 dataset which consists of 6000 32x32 pixel colour images with 10 classes. There are 5000 training images and 1000 test images.
- Data Preprocessing steps
  - Normalization
    The pixel values were normalized to have the mean of [0.486,0.456,0.406] and std of[0.229,0.224,0.225]
  -  Data augmentation was added to avoid overfitting, the following augmentations applied on the training set are random crop, random horizontal flip, random rotation and color jitter.
  -  All Images were converted to pytorch tensors using transforms.ToTensor()
- Data Splitting
  The data was  split into Train and Test

## Model Architecture 
ResNet 50 as the teacher model and MobilenetV2 as the student model, Since a pretrained model was used and most pretrained models are trained on ImageNet i modified the final layer of both models to accept CIFAR 10 10 classes. Same model was trained with the same learning rate, weight decay on the same number of epochs which is 40. The type of Evaluation metric used  is called Classification Accuracy.

## Results
I trained  the two models(Parent and Student) seperately to see how they performed on the dataset before training it with Knowledge Distillation. I got the Test Accuracy of 74.19% when i trained the ResNet50 on 40epochs and 71.19% on the Student model on the same number of epochs without distillation. When trained with distillation with a Temperature of 4 i got the accuracy of 75.78% which is a significant improvement.
  
  
  

  




- 


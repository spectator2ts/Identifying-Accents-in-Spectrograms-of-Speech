# Identifying-Accents-in-Spectrograms-of-Speech
This repository contains the code for the IASS(Identifying Accents in Spectrograms) project from 
[MPP Capstone Challenge](https://www.datasciencecapstone.org/competitions/16/identifying-accents-speech/page/49/).

## Problem statement
The goal is to predict the accent of the speaker from spectrograms of speech samples. A spectrogram is a visual representation of the various frequencies of sound as they vary with time. These spectrograms were generated from audio samples in the Mozilla Common Voice dataset. Each speech clip was sampled at 22,050 Hz, and contains an accent from one of the following three countries: Canada, India, and England.

## Programming Language
Python 3.7

## Modeling
Sequential neural network model was built for modeling. Data augmentation and normalization are implemented for preprosessing. Overlapping max pooling are also utilized. The model is supervised by the validation accuracy to verify improvement. 

The model contains 9 layers, including the max pooling layer. Filter size is (2 * 3). Activation functions are ReLu except for the last layer, where Softmax function is implemented for oupting the result. Adam is adopted as the optimizer and the loss is defined as categorical cross-entropy. The details are shown in the code provided.

## Result
After 150 epochs of training, the validation accuracy slightly fluctuate around 0.8. Thus I assume the true test accuracy may be around 0.78.

## Further Improvement
Below are the tricks I believe may improve the accuracy further but had not tried yet.

[Inception Net](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)

[Concatenate](https://keras.io/layers/merge/)

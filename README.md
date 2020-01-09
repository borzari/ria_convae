# RIA ConVAE Project

In this repository I'll upload everything related to my Research Internship Abroad (RIA) project. This internship started on 01/10/2019 and will end on 20/12/2019.

The goal is to build a Machine Learning (ML) based High Energy Physics (HEP) objects generator that is faster and more reliable than the state-of-the-art non-ML based generators that are most used nowadays. The main idea is to develop a Deep Convolutional Variational Autoencoder (ConVAE) to achieve that goal.

## First Steps

### Learning Deep Learning

The first step was to learn about Deep Learning (DL) by reading the book "Deep Learning" from Ian Goodfellow, Yoshua Bengio and Aaron Courville (http://www.deeplearningbook.org/).

### Standard MNIST Dataset

After that, a ConVAE for generating standard MNIST dataset images was built. Train and test errors are comparable. The generated images were in agreement with the MNIST train dataset.

### Sparse MNIST Dataset

A ConVAE for the sparse MNIST dataset, that is, what will be fed to the network is a matrix with the positions (x, y) and the intensities (I) of the 100 most intense pixels of the standard MNIST dataset, in intensity decreasing order was build and gave interesting results when comparing only the pixels positions. Something had to be done with respect to the intensities of the pixels.

### Superpixels MNIST dataset

To work with the intensities problem, the MNIST superpixels dataset was used. The pixels with intensity equal to 0 were removed from the dataset, and their intensities were ordered in decreasing order.

## Main Project

The description of the main project goes here.

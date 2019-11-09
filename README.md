# RIA ConVAE Project

In this repository I'll upload everything related to my Research Internship Abroad (RIA) project. This internship started on 01/10/2019 and will end on 20/12/2019.

The goal is to build a Machine Learning (ML) based High Energy Physics (HEP) objects generator that is faster and more reliable than the state-of-the-art non-ML based generators that are most used nowadays. The main idea is to develop a Deep Convolutional Variational Autoencoder (ConVAE) to achieve that goal.

## First Steps

### Learning Deep Learning

The first step was to learn about Deep Learning (DL) by reading the book "Deep Learning" from Ian Goodfellow, Yoshua Bengio and Aaron Courville (http://www.deeplearningbook.org/).

### Standard MNIST Dataset

After that, a ConVAE for generating standard MNIST dataset images was built. Train and test errors are comparable. The generated images were in agreement with the MNIST train dataset.

### Sparse MNIST Dataset

Right now, I'm working with a ConVAE for the sparse MNIST dataset, that is, what will be fed to the network is a matrix with the positions (x, y) and the intensities (I) of the 100 most intense pixels of the standard MNIST dataset, in intensity decreasing order.

## Main Project

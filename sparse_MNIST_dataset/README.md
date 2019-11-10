# Sparse MNIST Dataset

The sparse MNIST dataset has this name because it is represented as only the 100 most intense pixels from the MNIST dataset images. What is fed to the network is a 3x100 matrix with the position (x, y) and the intensity I of each pixel in intensity decreasing order.

This representation is closer to the representation of the data from the main project, since, as an example, in the calorimeters, the HEP objects are described by its energy deposited E, its angle \phi  and its pseudorapidity \eta .

The idea here was to test a different reconstruction loss term, since Mean Squared Error (MSE) would need to keep the order of the output pixels exactly like the input pixels. One solution was to test the Euclidean distance from the pixels in 3 different possibilities:

- getting the minimum of the Euclidean distances of each input pixel with one output pixel, and summing over the output pixels (this possibility will be referred as "oei");
- getting the minimum of the Euclidean distances of each output pixel with one input pixel, and summing over the input pixels (this possibility will be referred as "ieo");
- getting the symmetrized version of both approaches above (this possibility will be referred as "sym");

Also, an output pixel repulsion term was tested, since in the oei case above they tend to cluster into some regions of the image.

Two of those three possibilities worked very well for 50 epochs of training (~35 minutes without the repulsive term, ~75 minutes with the repulsive term), including in the image generation step: the sym case without a repulsive term, and the oei case with a repulsive term. The ieo case made some images with intense pixels all over the place.

This network is not optimized in any way (hyperparameters, network architecture, etc.)

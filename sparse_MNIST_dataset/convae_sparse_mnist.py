import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import time

# Starting time
start_time = time.time()

# Hyperparameters
num_epochs = 50
num_classes = 1
num_files = 600
batch_size = 100
learning_rate = 0.001
latent_dim = 10
beta = 1.0 # Repulsive term hyperparameter; if beta = 0.0, the repulsive term is 0.0
k = 0.5 # Repulsive term hyperparameter

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,2), stride=(1,2), padding=(0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,2), stride=(1,2), padding=(0))
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=(1,2), stride=(1,2), padding=(0))
        self.conv4 = nn.ConvTranspose2d(32, 1, kernel_size=(1,2), stride=(1,2), padding=(0))
        self.fc1 = nn.Linear(3 * 25 * 64, 2 * latent_dim)
        self.fc2 = nn.Linear(latent_dim, 3 * 25 * 64)

    def encode(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        mean = out[:,:latent_dim]
        logvar = 1e-6 + (out[:,latent_dim:])
        return mean, logvar

    def decode(self, z):
        out = self.fc2(z)
        out = torch.relu(out)
        out = out.view(batch_size, 64, 3, 25)
        out = self.conv3(out)
        out = torch.relu(out)
        out = self.conv4(out)
        out = 27 * torch.sigmoid(out) # The output x and y values should be between 0 and 27 (dimensions of a MNIST image)
        return out

    def reparameterize(self, mean, logvar):
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_decoded = model.decode(z)

    pdist = nn.PairwiseDistance(p=2) # Euclidean distance

    eucl = torch.zeros(batch_size).cuda()
    repulsive = torch.zeros(batch_size).cuda()

    # This removes the channel dimension from the tensor and makes pdist easier to use
    x_mod = torch.zeros(batch_size,3,100).cuda()
    x_mod = x[:,0,:,:]

    # This removes the channel dimension from the tensor and makes pdist easier to use
    x_decoded_mod = torch.zeros(batch_size,3,100).cuda()
    x_decoded_mod = x_decoded[:,0,:,:]

    for j in range(100):

        # Getting only one of the output pixels
        aux_x_decoded = torch.zeros(batch_size,3,1).cuda()
        aux_x_decoded[:,:,0] = x_decoded_mod[:,:,j]

        # Getting only one of the input pixels
        aux_x = torch.zeros(batch_size,3,1).cuda()
        aux_x[:,:,0] = x_mod[:,:,j]

        # This is comparing each pixel in the input with one pixel of the output, summing over the output pixels
#        eucl += torch.min((pdist(x_mod, aux_x_decoded)), dim = 1).values 

        # This is comparing each pixel in the output with one pixel of the input, summing over the input pixels
#        eucl += torch.min((pdist(x_decoded_mod, aux_x)), dim = 1).values 

        # This is the symmetrization of the two possibilities above
        eucl += 0.5 * torch.min((pdist(x_decoded_mod, aux_x)), dim = 1).values + 0.5 * torch.min((pdist(x_mod, aux_x_decoded)), dim = 1).values

        # Output pixels repulsive term. This works better for the first form of the euclidean distance term (comparing each pixel in the input with one pixel of the output)
        if (beta != 0.0):

            # This is getting all the output pixels that aren't aux_x_decoded
            aux_repulsive = torch.zeros(batch_size,2,99).cuda()
            aux_repulsive[:,:,:j] = x_decoded_mod[:,:2,:j]
            aux_repulsive[:,:,j:] = x_decoded_mod[:,:2,j+1:]

            repulsive += 1 / torch.exp((torch.min((pdist(aux_x_decoded[:,:2,:], aux_repulsive)), dim = 1).values) / k)

    eucl = - torch.sum(eucl) / batch_size
    repulsive = beta * (- torch.sum(repulsive) / batch_size)

    reconstruction_loss = eucl + repulsive

    KL_divergence = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians

    ELBO = reconstruction_loss - KL_divergence

    loss = - ELBO

    return loss, KL_divergence, -eucl, -repulsive

model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x_graph = np.arange(num_epochs)
y_eucl = np.arange(num_epochs)
y_kl = np.arange(num_epochs)
y_rep = np.arange(num_epochs)

loss_list = []

for epoch in range(num_epochs):

    kl_aux = 0.0
    eucl_aux = 0.0
    rep_aux = 0.0

    for y in range(num_files):

        # Load the train dataset
        input_train = torch.load('/afs/cern.ch/work/b/borzari/ml/sparse_MNIST_train_dataset_with_intensity/sparse_MNIST_'+str(y)+'.pt').cuda()

        # Train
        if epoch + 1 == num_epochs:
            output_train = model(input_train)
        loss, kl, eucl, rep = compute_loss(model, input_train)
        loss_list.append(loss.item())
        kl_aux += kl
        eucl_aux += eucl
        rep_aux += rep

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(epoch,loss.item())
    y_kl[epoch] = kl_aux / num_files
    y_eucl[epoch] = eucl_aux / num_files
    y_rep[epoch] = rep_aux / num_files

# Plot each component of the loss function
plt.figure()
plt.plot(x_graph, y_kl, label = "KL Divergence")
plt.plot(x_graph, y_eucl, label = 'Euclidean Distance')
plt.plot(x_graph, y_rep, label = 'Repulsive Term k=0.5, beta=1.0')
plt.xlabel('Epoch')
plt.ylabel('A. U.')
plt.title('Loss Function Components')
plt.legend()
plt.savefig('loss_components_graph.png')

# Save the model
torch.save(model.state_dict(), "model.pt")

sum = 0

for y in range(100):

    # Load the test dataset
    input_test = torch.load('/afs/cern.ch/work/b/borzari/ml/sparse_MNIST_test_dataset_with_intensity/sparse_MNIST_'+str(y)+'.pt').cuda()

    output_test = model(input_test)
    loss, kl, eucl, rep = compute_loss(model, input_test)

    sum += loss.item()

print('The average test loss is: ',sum/100)

output_train_image = torch.zeros(batch_size, 1, 28, 28).cuda()

# Transform the train output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(100):
        output_train_image[q, 0, int((output_train[q, 0, 0, k])), int((output_train[q, 0, 1, k]))] = output_train[q, 0, 2, k]

# Plot the train output images
fig = plt.figure(figsize=(10,10))

for o in range(batch_size):
    train_image = output_train_image.cpu().detach().numpy()
    plt.subplot(10, 10, o+1)
    plt.imshow(train_image[o, 0, :, :], cmap='gray')
    plt.colorbar()
    plt.axis('off')
plt.savefig('image_sparse_train_eucl_sym_100.png')

output_test_image = torch.zeros(batch_size, 1, 28, 28).cuda()

# Transform the test output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(100):
        output_test_image[q, 0, int((output_test[q, 0, 0, k])), int((output_test[q, 0, 1, k]))] = output_test[q, 0, 2, k]

fig = plt.figure(figsize=(10,10))

# Plot the test output images
for o in range(batch_size):
    test_image = output_test_image.cpu().detach().numpy()
    plt.subplot(10, 10, o+1)
    plt.imshow(test_image[o, 0, :, :], cmap='gray')
    plt.colorbar()
    plt.axis('off')
plt.savefig('image_sparse_test_eucl_sym_100.png')

# This will generate images from a vector of dimension latent_dim, where each entry is sampled from a standard gaussian (mean = 0.0 and stddev = 1.0)
z = torch.randn(batch_size, latent_dim).cuda()
output_gaussian = model.decode(z)
output_gaussian_image = torch.zeros(batch_size, 1, 28, 28).cuda()

# Transform the gaussian output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(100):
        output_gaussian_image[q, 0, int((output_gaussian[q, 0, 0, k])), int((output_gaussian[q, 0, 1, k]))] = output_gaussian[q, 0, 2, k]

# Plot the generated images
fig = plt.figure(figsize=(10,10))

for o in range(batch_size):
    gaussian_image = output_gaussian_image.cpu().detach().numpy()
    plt.subplot(10, 10, o+1)
    plt.imshow(gaussian_image[o, 0, :, :], cmap='gray')
    plt.colorbar()
    plt.axis('off')
plt.savefig('image_sparse_gaussian_eucl_sym_100.png')

# Ending time
end_time = time.time()

print("The total time is ",((end_time-start_time)/60.0)," minutes.")

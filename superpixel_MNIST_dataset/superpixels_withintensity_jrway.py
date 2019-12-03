import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Starting time
start_time = time.time()

# Hyperparameters
num_epochs = 500
num_classes = 1
batch_size = 100
learning_rate = 0.001
latent_dim = 10
beta = 27.0 * 27.0 # Beta controls the magnitude of the intensity terms in the loss function (beta = 27 * 27 is the equivalent of multiplying the input intensities by 27)
drop_prob = 0.0 # Probability to keep a node in the dropout layer (not being used here, but might be important in the future)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1,5), stride=(1), padding=(0,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,5), stride=(1), padding=(0,1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1), padding=(0,1))
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(1,5), stride=(1), padding=(0,1))
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=(1,5), stride=(1), padding=(0,1))
        self.conv6 = nn.ConvTranspose2d(16, 1, kernel_size=(1,5), stride=(1), padding=(0,1))
        self.fc1 = nn.Linear(3 * 19 * 64, 1500)
        self.fc2 = nn.Linear(1500, 2 * latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1500)
        self.fc4 = nn.Linear(1500, 3 * 19 * 64)

        self.drop = nn.Dropout(drop_prob)

    def encode(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = torch.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        mean = out[:,:latent_dim]
        logvar = 1e-6 + (out[:,latent_dim:])
        return mean, logvar

    def decode(self, z):
        out = self.fc3(z)
        out = torch.relu(out)
        out = self.fc4(out)
        out = torch.relu(out)
        out = out.view(batch_size, 64, 3, 19)
        out = self.conv4(out)
        out = torch.relu(out)
        out = self.conv5(out)
        out = torch.relu(out)
        out = self.conv6(out)
        out[:, :, :2] = 27 * torch.sigmoid(out[:, :, :2]) # The output x and y values should be between 0 and 27 (dimensions of a MNIST image)

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

    # This removes the channel dimension from the tensor and makes pdist easier to use
    x_pos = torch.zeros(batch_size,2,25).cuda() # Input pixels x and y positions
    x_pos = x[:,0,:2,:]
    x_int = torch.zeros(batch_size,25).cuda() # Input pixels intensities
    x_int = x[:,0,2,:]

    x_pos = x_pos.view(100, 2, 1, 25) # This step is important for the calculation of dist

    # This removes the channel dimension from the tensor and makes pdist easier to use
    x_decoded_pos = torch.zeros(batch_size,2,25).cuda() # Output pixels x and y positions
    x_decoded_pos = x_decoded[:,0,:2,:]
    x_decoded_int = torch.zeros(batch_size,25).cuda() # Output pixels intensities
    x_decoded_int[:] = x_decoded[:,0,2,:]

    x_decoded_pos = x_decoded_pos.view(batch_size, 2, 25, 1) # This and the next line are also important steps for the calculation of dist
    x_decoded_pos = torch.repeat_interleave(x_decoded_pos, 25, -1)

    dist = torch.pow(pdist(x_pos, x_decoded_pos),2) # Using the transformed tensors above, this will calculate the squared euclidean distance between every pair of pixel in the input and output tensors (it will have dimension 100, 25, 25 that is, 100 images per batch, with 25 pixels each)

    ieo = torch.min(dist, dim = 1) # ieo = input each output; this is the min between the distances of the pixels, but fixing the input pixel being compared
    ieo_idx = ieo.indices.clone() # This is a subtlety to keep the loss function differentiable

    oei = torch.min(dist, dim = 2) # oei = output each input; this is the min between the distances of the pixels, but fixing the output pixel being compared
    oei_idx = oei.indices.clone() # This is a subtlety to keep the loss function differentiable

    aux_idx = 25 * torch.arange(100).cuda() # These auxiliary indices will help using the torch.take() function
    aux_idx = aux_idx.view(100, 1)
    aux_idx = torch.repeat_interleave(aux_idx, 25, axis=-1)

    ieo_idx = ieo_idx + aux_idx
    oei_idx = oei_idx + aux_idx

    get_x_int = torch.take(x_int, oei_idx) # The input tensor is unrolled, and the result will have the shape of oei_idx. But, the indices of oei_idx must match the positions in the unrolled x_int. That is why aux_idx is being used

    get_x_decoded_int = torch.take(x_decoded_int,ieo_idx)

    eucl = ieo.values + oei.values + beta * (torch.pow((x_decoded_int - get_x_int), 2) + torch.pow((x_int - get_x_decoded_int), 2)) # This is summing the distances between the pixels, first by fixing the input pixels, then by fixing the output pixels, and the difference between their intensities

    eucl = torch.sum(eucl) / batch_size # This is important to get the error per image

    reconstruction_loss = - eucl # This step is conceptually important when dealing with the ELBO loss function

    KL_divergence = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians

    ELBO = reconstruction_loss - KL_divergence # The ELBO loss function is easier to understand when written on this form

    loss = - ELBO # We want to minimaze the loss function

    return loss, KL_divergence, eucl

model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer is one of the most used optimizer

# The following lists will be used in the plots of the graph containing the components of the loss function

x_graph = []

tr_y_rec = []
tr_y_kl = []
tr_y_loss = []

val_y_rec = []
val_y_kl = []
val_y_loss = []

# Loading the datasets

train_dataset = torch.load('MNISTsuperpixel_train_withoutzeros_onlyeight.pt')
valid_dataset = torch.load('MNISTsuperpixel_valid_withoutzeros_onlyeight.pt')
test_dataset = torch.load('MNISTsuperpixel_test_withoutzeros_onlyeight.pt')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

min_loss, stale_epochs = 999999.0, 0 # These quantities will be used in the early stopping part of the model

for epoch in range(num_epochs):

    x_graph.append(epoch)

    # Some auxiliary variables

    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0

    val_loss_aux = 0.0
    val_kl_aux = 0.0
    val_rec_aux = 0.0

    for y, (images_train, labels_train) in enumerate(train_loader): # Getting the images and their labels (in this case there is only the digit 8)

        if y == (len(train_loader) - 1): # Stop before the end of the train_loader since the total number of 8 images is not divisible by 100
            break

        input_train = images_train[:, :, :].cuda()
        input_train[:, :, 2, :] = 27 * input_train[:, :, 2, :] # Maybe, dividing the positions by 27 would also work, but the reconstructed images do not look like the input images

        # Training
        output_train = model(input_train)
        tr_loss, tr_kl, tr_eucl = compute_loss(model, input_train)
        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

    tr_y_loss.append(tr_loss_aux.item()/(len(train_loader) - 1))
    tr_y_kl.append(tr_kl_aux.item()/(len(train_loader) - 1))
    tr_y_rec.append(tr_rec_aux.item()/(len(train_loader) - 1))

    # This is the validation dataset calculations (it doesn't require grad because the model will not learn from this data)

    with torch.no_grad():

        for w, (images_valid, labels_valid) in enumerate(valid_loader):

            if w == (len(valid_loader) - 1):
                break

            input_valid = images_valid[:, :, :].cuda()
            input_valid[:, :, 2, :] = 27 * input_valid[:, :, 2, :]

            val_loss, val_kl, val_eucl = compute_loss(model, input_valid)
            val_loss_aux += val_loss
            val_kl_aux += val_kl
            val_rec_aux += val_eucl

    val_y_loss.append(val_loss_aux.item()/(len(valid_loader) - 1))
    val_y_kl.append(val_kl_aux.item()/(len(valid_loader) - 1))
    val_y_rec.append(val_rec_aux.item()/(len(valid_loader) - 1))

    # Early stopping part of the code

    if stale_epochs > 20:
        print("Early stopped")
        break

    if val_loss_aux.item()/(len(valid_loader) - 1) < min_loss:
        min_loss = val_loss_aux.item()/(len(valid_loader) - 1)
        stale_epochs = 0
    else:
        stale_epochs += 1

    print('Epoch: {} -- Train loss: {} -- Validation loss: {}'.format(epoch, tr_loss_aux.item()/(len(train_loader) - 1), val_loss_aux.item()/(len(valid_loader) - 1)))

# Plot each component of the loss function
plt.figure()
plt.plot(x_graph, tr_y_kl, label = "Train KL Divergence")
plt.plot(x_graph, tr_y_rec, label = 'Train Reconstruction Loss')
plt.plot(x_graph, tr_y_loss, label = 'Train Total Loss')
plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence")
plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss')
plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('A. U.')
plt.title('Loss Function Components')
plt.legend()
plt.savefig('loss_superpixel_jrway_27int_beta'+str(int(beta))+'_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

# Save the model
torch.save(model.state_dict(), 'model_superpixel_jrway_27int_beta'+str(int(beta))+'_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.pt')

# Testing the model

sum = 0

for y, (images_test, labels_test) in enumerate(test_loader):

    if y == (len(test_loader) - 1):
        break

    input_test = images_test[:, :, :].cuda()
    input_test[:, :, 2, :] = 27 * input_test[:, :, 2, :]

    output_test = model(input_test)
    loss, kl, eucl = compute_loss(model, input_test) # kl and eucl are not being used here

    sum += loss.item()

print('The average test loss is: ',sum/(len(test_loader) - 1))

MNIST_train_image = torch.zeros(batch_size, 1, 28, 28).cuda()

input_train[:,:,:2,:] = torch.round(input_train[:,:,:2,:])

# Everything in the following is for plotting the images (it will plot 100 images of the MNIST superpixel train dataset and test dataset, the output train dataset, test dataset and the images generated using the standard gaussians)

# Transform the train output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(25):

        MNIST_train_image[q, 0, int((input_train[q, 0, 0, k])), int((input_train[q, 0, 1, k]))] = input_train[q, 0, 2, k]

fig, (axs) = plt.subplots(10, 10)
fig.set_size_inches(15.5, 13.5)
fig.subplots_adjust(right=0.8)
plt.subplots_adjust(wspace=0.5)
k = 0
train_input_image = MNIST_train_image.cpu().detach().numpy()
for i in range(10):
    for j in range(10):
        img = axs[i, j].imshow(train_input_image[k, 0, :, :], cmap='binary')
        divider = make_axes_locatable(axs[i, j])
        axs[i, j].xaxis.set_visible(False)
        plt.setp(axs[i, j].get_yticklabels(), visible=False)
        axs[i, j].tick_params(axis='both', which='both', length=0)
        k += 1
cbar_ax = fig.add_axes([0.85, 0.115, 0.02, 0.76])
fig.colorbar(img, cax=cbar_ax)
fig.savefig('image_MNISTsuperpixel_jrway_train_27int_beta'+str(int(beta))+'_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

MNIST_test_image = torch.zeros(batch_size, 1, 28, 28).cuda()

input_test[:,:,:2,:] = torch.round(input_test[:,:,:2,:])

# Transform the train output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(25):

        MNIST_test_image[q, 0, int((input_test[q, 0, 0, k])), int((input_test[q, 0, 1, k]))] = input_test[q, 0, 2, k]

fig, (axs) = plt.subplots(10, 10)
fig.set_size_inches(15.5, 13.5)
fig.subplots_adjust(right=0.8)
plt.subplots_adjust(wspace=0.5)
k = 0
test_input_image = MNIST_test_image.cpu().detach().numpy()
for i in range(10):
    for j in range(10):
        img = axs[i, j].imshow(test_input_image[k, 0, :, :], cmap='binary')
        divider = make_axes_locatable(axs[i, j])
        axs[i, j].xaxis.set_visible(False)
        plt.setp(axs[i, j].get_yticklabels(), visible=False)
        axs[i, j].tick_params(axis='both', which='both', length=0)
        k += 1
cbar_ax = fig.add_axes([0.85, 0.115, 0.02, 0.76])
fig.colorbar(img, cax=cbar_ax)
fig.savefig('image_MNISTsuperpixel_jrway_test_27int_beta'+str(int(beta))+'_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

output_train_image = torch.zeros(batch_size, 1, 28, 28).cuda()

output_train[:,:,:2,:] = torch.round(output_train[:,:,:2,:])

# Transform the train output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(25):

        output_train_image[q, 0, int((output_train[q, 0, 0, k])), int((output_train[q, 0, 1, k]))] = output_train[q, 0, 2, k]

fig, (axs) = plt.subplots(10, 10)
fig.set_size_inches(15.5, 13.5)
fig.subplots_adjust(right=0.8)
plt.subplots_adjust(wspace=0.5)
k = 0
train_image = output_train_image.cpu().detach().numpy()
for i in range(10):
    for j in range(10):
        img = axs[i, j].imshow(train_image[k, 0, :, :], cmap='binary')
        divider = make_axes_locatable(axs[i, j])
        axs[i, j].xaxis.set_visible(False)
        plt.setp(axs[i, j].get_yticklabels(), visible=False)
        axs[i, j].tick_params(axis='both', which='both', length=0)
        k += 1
cbar_ax = fig.add_axes([0.85, 0.115, 0.02, 0.76])
fig.colorbar(img, cax=cbar_ax)
fig.savefig('image_superpixel_jrway_train_27int_beta'+str(int(beta))+'_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

output_test_image = torch.zeros(batch_size, 1, 28, 28).cuda()

output_test[:,:,:2,:] = torch.round(output_test[:,:,:2,:])

# Transform the test output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(25):
        output_test_image[q, 0, int((output_test[q, 0, 0, k])), int((output_test[q, 0, 1, k]))] = output_test[q, 0, 2, k]

fig, (axs) = plt.subplots(10, 10)
fig.set_size_inches(15.5, 13.5)
fig.subplots_adjust(right=0.8)
plt.subplots_adjust(wspace=0.5)
k = 0
test_image = output_test_image.cpu().detach().numpy()
for i in range(10):
    for j in range(10):
        img = axs[i, j].imshow(test_image[k, 0, :, :], cmap='binary')
        divider = make_axes_locatable(axs[i, j])
        axs[i, j].xaxis.set_visible(False)
        plt.setp(axs[i, j].get_yticklabels(), visible=False)
        axs[i, j].tick_params(axis='both', which='both', length=0)
        k += 1
cbar_ax = fig.add_axes([0.85, 0.115, 0.02, 0.76])
fig.colorbar(img, cax=cbar_ax)
fig.savefig('image_superpixel_jrway_test_27int_beta'+str(int(beta))+'_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

# This will generate images from a vector of dimension latent_dim, where each entry is sampled from a standard gaussian (mean = 0.0 and stddev = 1.0)
z = torch.randn(batch_size, latent_dim).cuda()
output_gaussian = model.decode(z)
output_gaussian[:,:,:2,:] = torch.round(output_gaussian[:,:,:2,:])

output_gaussian_image = torch.zeros(batch_size, 1, 28, 28).cuda()

# Transform the gaussian output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(25):
        output_gaussian_image[q, 0, int((output_gaussian[q, 0, 0, k])), int((output_gaussian[q, 0, 1, k]))] = output_gaussian[q, 0, 2, k]

fig, (axs) = plt.subplots(10, 10)
fig.set_size_inches(15.5, 13.5)
fig.subplots_adjust(right=0.8)
plt.subplots_adjust(wspace=0.5)
k = 0
gaussian_image = output_gaussian_image.cpu().detach().numpy()
for i in range(10):
    for j in range(10):
        img = axs[i, j].imshow(gaussian_image[k, 0, :, :], cmap='binary')
        divider = make_axes_locatable(axs[i, j])
        axs[i, j].xaxis.set_visible(False)
        plt.setp(axs[i, j].get_yticklabels(), visible=False)
        axs[i, j].tick_params(axis='both', which='both', length=0)
        k += 1
cbar_ax = fig.add_axes([0.85, 0.115, 0.02, 0.76])
fig.colorbar(img, cax=cbar_ax)
fig.savefig('image_superpixel_jrway_gaussian_27int_beta'+str(int(beta))+'_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

# Ending time
end_time = time.time()

print("The total time is ",((end_time-start_time)/60.0)," minutes.")

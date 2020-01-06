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
num_pixels = 25
batch_size = 100
learning_rate = 0.001
latent_dim = 10

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,5), stride=(1), padding=(0))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv6 = nn.ConvTranspose2d(16, 1, kernel_size=(3,5), stride=(1), padding=(0))
        self.fc1 = nn.Linear(1 * (num_pixels - 12) * 64, 1500)
        self.fc2 = nn.Linear(1500, 2 * latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1500)
        self.fc4 = nn.Linear(1500, 1 * (num_pixels - 12) * 64)

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
        out = out.view(batch_size, 64, 1, (num_pixels - 12))
        out = self.conv4(out)
        out = torch.relu(out)
        out = self.conv5(out)
        out = torch.relu(out)
        out = self.conv6(out)
        out = 27 * torch.sigmoid(out) # Every feature is in the range [0, 27]

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

    x_pos = torch.zeros(batch_size,3,num_pixels).cuda()
    x_pos = x[:,0,:,:] # Removes the channel dimension to make the following calculations easier

    x_pos = x_pos.view(batch_size, 3, 1, num_pixels) # Changes the dimension of the tensor so that dist is the distance between every pair of input and output pixels

    x_decoded_pos = torch.zeros(batch_size,3,num_pixels).cuda()
    x_decoded_pos = x_decoded[:,0,:,:] # Removes the channel dimension to make the following calculations easier

    x_decoded_pos = x_decoded_pos.view(batch_size, 3, num_pixels, 1) # Changes the dimension of the tensor so that dist is the distance between every pair of input and output pixels
    x_decoded_pos = torch.repeat_interleave(x_decoded_pos, num_pixels, -1) 

    dist = torch.pow(pdist(x_pos, x_decoded_pos),2)

    ieo = torch.min(dist, dim = 1) # Gets the value of the distance between the closest output pixels to all the input pixels of the images in a batch (all features of the pixels)

    oei = torch.min(dist, dim = 2) # Gets the value of the distance between the closest input pixels to all the input pixels of the images in a batch (all features of the pixels)

    eucl = ieo.values + oei.values # Symmetrical euclidean distances

    eucl = torch.sum(eucl) / batch_size # Average symmetrical euclidean distance per image

    reconstruction_loss = - eucl

    KL_divergence = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians

    ELBO = reconstruction_loss - KL_divergence

    loss = - ELBO

    return loss, KL_divergence, eucl

model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x_graph = []

tr_y_rec = []
tr_y_kl = []
tr_y_loss = []

val_y_rec = []
val_y_kl = []
val_y_loss = []

train_dataset = torch.load('MNISTsuperpixel_train_withoutzeros_onlyeight.pt')
valid_dataset = torch.load('MNISTsuperpixel_valid_withoutzeros_onlyeight.pt')
test_dataset = torch.load('MNISTsuperpixel_test_withoutzeros_onlyeight.pt')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

min_loss, stale_epochs = 999999.0, 0 # Early stopping

for epoch in range(num_epochs):

    x_graph.append(epoch)

    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0

    val_loss_aux = 0.0
    val_kl_aux = 0.0
    val_rec_aux = 0.0

    for y, (images_train, labels_train) in enumerate(train_loader):

        if y == (len(train_loader) - 1):
            break

        input_train = images_train[:, :, :].cuda()
        input_train[:, :, 2, :] = 27 * input_train[:, :, 2, :] # Multiply the intensity of the pixels by 27

        # Train
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
#plt.savefig('loss_superpixel_euclsquared_thirdaxis_alltensorloss_plotloss_27sig_27int_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')
#plt.savefig('paper_super_loss_thirdaxis.png')

# Save the model
#torch.save(model.state_dict(), 'model_superpixel_euclsquared_thirdaxis_alltensorloss_plotloss_27sig_27int_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.pt')
#torch.save(model.state_dict(), 'paper_super_model_thirdaxis.pt')

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

input_train_image = torch.zeros(batch_size, 1, 28, 28).cuda()

input_train[:,:,:2,:] = torch.round(input_train[:,:,:2,:])

# Transform the train output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(25):

        input_train_image[q, 0, int((input_train[q, 0, 0, k])), int((input_train[q, 0, 1, k]))] = input_train[q, 0, 2, k]

fig, (axs) = plt.subplots(10, 10)
fig.set_size_inches(15.5, 13.5)
fig.subplots_adjust(right=0.8)
plt.subplots_adjust(wspace=0.5)
k = 0
train_input_image = input_train_image.cpu().detach().numpy()
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
#fig.savefig('image_MNISTsuperpixel_train_euclsquared_thirdaxis_alltensorloss_plotloss_27sig_27int_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')
#fig.savefig('paper_mnist_train.png')

input_test_image = torch.zeros(batch_size, 1, 28, 28).cuda()

input_test[:,:,:2,:] = torch.round(input_test[:,:,:2,:])

# Transform the train output 3x100 matrices into 28x28 matrices
for q in range(batch_size):
    for k in range(25):

        input_test_image[q, 0, int((input_test[q, 0, 0, k])), int((input_test[q, 0, 1, k]))] = input_test[q, 0, 2, k]

fig, (axs) = plt.subplots(10, 10)
fig.set_size_inches(15.5, 13.5)
fig.subplots_adjust(right=0.8)
plt.subplots_adjust(wspace=0.5)
k = 0
test_input_image = input_test_image.cpu().detach().numpy()
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
#fig.savefig('image_MNISTsuperpixel_test_euclsquared_thirdaxis_alltensorloss_plotloss_27sig_27int_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')
#fig.savefig('paper_mnist_test.png')

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
#fig.savefig('image_superpixel_train_euclsquared_thirdaxis_alltensorloss_plotloss_27sig_27int_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')
#fig.savefig('paper_network_train.png')

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
#fig.savefig('paper_network_test.png')

#fig.savefig('image_superpixel_test_euclsquared_thirdaxis_alltensorloss_plotloss_27sig_27int_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

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
#fig.savefig('paper_network_gaussian.png')
#fig.savefig('image_superpixel_gaussian_euclsquared_thirdaxis_alltensorloss_plotloss_27sig_27int_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

# Ending time
end_time = time.time()

print("The total time is ",((end_time-start_time)/60.0)," minutes.")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt

# Start time
start_time = time.time()

# Hyperparameters
num_epochs = 60
num_classes = 1
batch_size = 100
latent_dim = 50
learning_rate = 0.0001

DATA_PATH = 'C:\\Users\Andy\PycharmProjects\MNISTData'
MODEL_STORE_PATH = 'C:\\Users\Andy\PycharmProjects\pytorch_models\\'

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor()])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("Files loaded!")

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 2 * latent_dim)
        self.fc2 = nn.Linear(latent_dim, 7 * 7 * 64)

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
        out = out.view(batch_size,64,7,7)
        out = self.conv3(out)
        out = torch.relu(out)
        out = self.conv4(out)
#        out = torch.selu(out)
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

    reconstruction_loss = (- torch.sum(torch.pow((x - x_decoded), 2))) / batch_size # Mean Squared Error

    KL_divergence = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians

    ELBO = reconstruction_loss - KL_divergence

    loss = -ELBO

    return loss

model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = images.cuda()

        # Run the forward pass
        if (epoch + 1) == num_epochs:
            if (i + 1) % (6 * batch_size) == 0:
                outputs = model(images)
        loss = compute_loss(model, images)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % (3 * batch_size) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

total_step = len(test_loader)
loss_list_test = []

for p in range(1):

    for i, (imagest, labelst) in enumerate(test_loader):

        imagest = imagest.cuda()

        # Run the forward pass
        outputst = model(imagest)
        losst = compute_loss(model, imagest)
        loss_list_test.append(losst.item())

        if (i + 1) % batch_size == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(p + 1, 1, i + 1, total_step, losst.item()))

fig = plt.figure(figsize=(10,10))

for k in range(batch_size):

    train_image = outputs.cpu()
    train_image = train_image.detach()
    train_image = train_image.numpy()
    plt.subplot(10, 10, k+1)
    plt.imshow(train_image[k, 0, :, :], cmap = 'gray')
    plt.colorbar()
    plt.axis('off')

plt.savefig('image_train_'+str(batch_size)+'.png') # ConVAE train image

fig = plt.figure(figsize=(10,10))

for l in range(batch_size):

    test_image = outputst.cpu()
    test_image = test_image.detach()
    test_image = test_image.numpy()
    plt.subplot(10, 10, l+1)
    plt.imshow(test_image[l, 0, :, :], cmap = 'gray')
    plt.colorbar()
    plt.axis('off')

plt.savefig('image_test_'+str(batch_size)+'.png') # ConVAE test image

fig = plt.figure(figsize=(10,10))

for j in range(batch_size):

    mnist_image = images.cpu()
    mnist_image = mnist_image.detach()
    mnist_image = mnist_image.numpy()
    plt.subplot(10, 10, j+1)
    plt.imshow(mnist_image[j, 0, :, :], cmap = 'gray')
    plt.colorbar()
    plt.axis('off')

plt.savefig('image_mnist_train_'+str(batch_size)+'.png') # MNIST dataset train image

fig = plt.figure(figsize=(10,10))

for m in range(batch_size):

    mnist_image = imagest.cpu()
    mnist_image = mnist_image.detach()
    mnist_image = mnist_image.numpy()
    plt.subplot(10, 10, m+1)
    plt.imshow(mnist_image[m, 0, :, :], cmap = 'gray')
    plt.colorbar()
    plt.axis('off')

plt.savefig('image_mnist_test_'+str(batch_size)+'.png') # MNIST dataset test image

fig = plt.figure(figsize=(10,10))

z = torch.randn(batch_size,latent_dim)
z = z.cuda()
out = model.decode(z)

fig = plt.figure(figsize=(10,10))

for o in range(batch_size):

    gaussian_image = out.cpu()
    gaussian_image = gaussian_image.detach()
    gaussian_image = gaussian_image.numpy()
    plt.subplot(10, 10, o+1)
    plt.imshow(gaussian_image[o, 0, :, :], cmap = 'gray')
    plt.colorbar()
    plt.axis('off')

plt.savefig('image_gaussians_'+str(batch_size)+'.png') # ConVAE gaussians image

# End time
end_time = time.time()

# Print the time
print("It took ",(end_time - start_time)/60.0," minutes to run everything.")

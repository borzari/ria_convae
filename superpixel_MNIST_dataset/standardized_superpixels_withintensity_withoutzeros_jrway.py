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
num_epochs = 50
num_classes = 1
batch_size = 100
learning_rate = 0.001
latent_dim = 10
beta_seq = [0.03, 0.1, 0.3, 1.0] # KL divergence factor in the loss function

for beta in beta_seq:

    print('beta is ',beta)

    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()

            self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,5), stride=(1), padding=(0))
            self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,5), stride=(1), padding=(0))
            self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1), padding=(0))
            self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(1,5), stride=(1), padding=(0))
            self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=(1,5), stride=(1), padding=(0))
            self.conv6 = nn.ConvTranspose2d(16, 1, kernel_size=(3,5), stride=(1), padding=(0))
            self.fc1 = nn.Linear(1 * 13 * 64, 1500)
            self.fc2 = nn.Linear(1500, 2 * latent_dim)
            self.fc3 = nn.Linear(latent_dim, 1500)
            self.fc4 = nn.Linear(1500, 1 * 13 * 64)
            self.m1 = nn.Hardtanh(-2.1349, 2.0604, inplace=False) # Min and max values of the standardized x positions
            self.m2 = nn.Hardtanh(-3.1659, 3.2496, inplace=False) # Min and max values of the standardized y positions
            self.m3 = nn.Hardtanh(-3.7453, 1.9585, inplace=False) # Min and max values of the standardized intensities
    
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
            out = out.view(batch_size, 64, 1, 13)
            out = self.conv4(out)
            out = torch.relu(out)
            out = self.conv5(out)
            out = torch.relu(out)
            out = self.conv6(out)
    #        out = 27 * torch.sigmoid(out) # Every feature is in the range [0, 27]
            out[:,0,0,:] = self.m1(out[:,0,0,:].clone())
            out[:,0,1,:] = self.m2(out[:,0,1,:].clone())
            out[:,0,2,:] = self.m3(out[:,0,2,:].clone())
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
    
        x_pos = torch.zeros(batch_size,2,25).cuda()
        x_pos = x[:,0,:2,:] # Removes the channel dimension to make the following calculations easier
        x_int = torch.zeros(batch_size,25).cuda()
        x_int = x[:,0,2,:] # Removes the channel dimension to make the following calculations easier
    
        x_pos = x_pos.view(100, 2, 1, 25) # Changes the dimension of the tensor so that dist is the distance between every pair of input and output pixels
    
        x_decoded_pos = torch.zeros(batch_size,2,25).cuda()
        x_decoded_pos = x_decoded[:,0,:2,:] # Removes the channel dimension to make the following calculations easier
        x_decoded_int = torch.zeros(batch_size,25).cuda()
        x_decoded_int = x_decoded[:,0,2,:] # Removes the channel dimension to make the following calculations easier
    
        x_decoded_pos = x_decoded_pos.view(batch_size, 2, 25, 1) # Changes the dimension of the tensor so that dist is the distance between every pair of input and output pixels
        x_decoded_pos = torch.repeat_interleave(x_decoded_pos, 25, -1)
    
        dist = torch.pow(pdist(x_pos, x_decoded_pos),2)
    
        ieo = torch.min(dist, dim = 1) # Gets the value of the distance between the closest output pixels to all the input pixels of the images in a batch (only pixels positions)
        ieo_idx = ieo.indices.clone() # Necessary to avoid pytorch errors
    
        oei = torch.min(dist, dim = 2) # Gets the value of the distance between the closest input pixels to all the input pixels of the images in a batch (only pixels positions)
        oei_idx = oei.indices.clone() # Necessary to avoid pytorch errors
    
        aux_idx = 25 * torch.arange(100).cuda() # This will help getting the right pixel intensities
        aux_idx = aux_idx.view(100, 1)
        aux_idx = torch.repeat_interleave(aux_idx, 25, axis=-1)
    
        ieo_idx = ieo_idx + aux_idx
        oei_idx = oei_idx + aux_idx
    
        get_x_int = torch.take(x_int, oei_idx) # Get the intensity of the closest input pixel to a given output pixel
    
        get_x_decoded_int = torch.take(x_decoded_int,ieo_idx) # Get the intensity of the closest output pixel to a given input pixel
    
        eucl = ieo.values + oei.values + (torch.pow((x_decoded_int - get_x_int), 2) + torch.pow((x_int - get_x_decoded_int), 2)) # Symmetrical euclidean distances
    
        eucl = torch.sum(eucl) / batch_size # Average symmetrical euclidean distance per image
    
        reconstruction_loss = - eucl
    
        KL_divergence = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians
    
        ELBO = reconstruction_loss - (beta * KL_divergence)
    
        loss = - ELBO
    
        return loss, (beta * KL_divergence), eucl
    
    model = ConvNet()
    model = model.cuda()
    
    torch.autograd.set_detect_anomaly(True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    x_graph = []
    
    tr_y_rec = []
    tr_y_kl = []
    tr_y_loss = []
    
    su_jr_teinpx = torch.Tensor().cuda()
    su_jr_teoutx = torch.Tensor().cuda()
    su_jr_gaussx = torch.Tensor().cuda()
    su_jr_teinpy = torch.Tensor().cuda()
    su_jr_teouty = torch.Tensor().cuda()
    su_jr_gaussy = torch.Tensor().cuda()
    su_jr_teinpi = torch.Tensor().cuda()
    su_jr_teouti = torch.Tensor().cuda()
    su_jr_gaussi = torch.Tensor().cuda()
    
    train_mean0 = 13.2133
    train_std0 = 5.7207
    test_mean0 = 13.2297
    test_std0 = 5.7175
    train_mean1 = 13.3371
    train_std1 = 3.8968
    test_mean1 = 13.3451
    test_std1 = 3.7948
    train_mean2 = 0.6580
    train_std2 = 0.1746
    test_mean2 = 0.6621
    test_std2 = 0.0800
    
    train_dataset = torch.load('MNISTsuperpixel_train_withoutzeros_onlyeight_std.pt')
    test_dataset = torch.load('MNISTsuperpixel_test_withoutzeros_onlyeight_std.pt')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
    
        x_graph.append(epoch)
    
        tr_loss_aux = 0.0
        tr_kl_aux = 0.0
        tr_rec_aux = 0.0
    
        for y, (images_train) in enumerate(train_loader):
    
            if y == (len(train_loader) - 1):
                break
    
            input_train = images_train[:, :, :].cuda()
    #        input_train[:, :, 2, :] = 27 * input_train[:, :, 2, :] # Multiply the intensity of the pixels by 27
    
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
    
        print('Epoch: {} -- Train loss: {}'.format(epoch, tr_loss_aux.item()/(len(train_loader) - 1)))
    
    
    # Plot each component of the loss function
    plt.figure()
    plt.plot(x_graph, tr_y_kl, label = "Train KL Divergence")
    plt.plot(x_graph, tr_y_rec, label = 'Train Reconstruction Loss')
    plt.plot(x_graph, tr_y_loss, label = 'Train Total Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('A. U.')
    plt.title('Loss Function Components')
    plt.legend()
    plt.savefig('su_jr_loss_std_kl'+str(int(100 * beta))+'_hardtanh.jpg')
    
    # Save the model
    torch.save(model.state_dict(), 'su_jr_model_std_kl'+str(int(100 * beta))+'_hardtanh.pt')
    
    sum = 0
    
    for y, (images_test) in enumerate(test_loader):
    
        if y == (len(test_loader) - 1):
            break
    
        input_test = images_test[:, :, :].cuda()
    #    input_test[:, :, 2, :] = 27 * input_test[:, :, 2, :]
    
        output_test = model(input_test)
        loss, kl, eucl = compute_loss(model, input_test) # kl and eucl are not being used here
    
        z = torch.randn(batch_size, latent_dim).cuda()
        output_gaussian = model.decode(z)
    
        for j in range(batch_size):

            su_ta_teinpx = torch.cat((su_ta_teinpx, torch.round((input_test[j,0,0]*test_std0)+test_mean0))) # Inverse standardization transformation applied in input test x position values
            su_ta_teoutx = torch.cat((su_ta_teoutx, torch.round((output_test[j,0,0]*train_std0)+train_mean0))) # Inverse standardization transformation applied in output test x position values
            su_ta_gaussx = torch.cat((su_ta_gaussx, torch.round((output_gaussian[j,0,0]*train_std0)+train_mean0))) # Inverse standardization transformation applied in generated x position values
            su_ta_teinpy = torch.cat((su_ta_teinpy, torch.round((input_test[j,0,1]*test_std1)+test_mean1))) # Inverse standardization transformation applied in input test y position values
            su_ta_teouty = torch.cat((su_ta_teouty, torch.round((output_test[j,0,1]*train_std1)+train_mean1))) # Inverse standardization transformation applied in output test y position values
            su_ta_gaussy = torch.cat((su_ta_gaussy, torch.round((output_gaussian[j,0,1]*train_std1)+train_mean1))) # Inverse standardization transformation applied in generated y position values
            su_ta_teinpi = torch.cat((su_ta_teinpi, ((input_test[j,0,2]*test_std2)+test_mean2))) # Inverse standardization transformation applied in input test intensity values
            su_ta_teouti = torch.cat((su_ta_teouti, ((output_test[j,0,2]*train_std2)+train_mean2))) # Inverse standardization transformation applied in output test intensity values
            su_ta_gaussi = torch.cat((su_ta_gaussi, ((output_gaussian[j,0,2]*train_std2)+train_mean2))) # Inverse standardization transformation applied in generated intensity values
    
        sum += loss.item()
    
    print('The average test loss is: ',sum/(len(test_loader) - 1))
    
    torch.save(su_jr_teinpx.cpu().detach(), 'su_jr_std_teinpx_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_teoutx.cpu().detach(), 'su_jr_std_teoutx_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_gaussx.cpu().detach(), 'su_jr_std_gaussx_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_teinpy.cpu().detach(), 'su_jr_std_teinpy_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_teouty.cpu().detach(), 'su_jr_std_teouty_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_gaussy.cpu().detach(), 'su_jr_std_gaussy_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_teinpi.cpu().detach(), 'su_jr_std_teinpi_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_teouti.cpu().detach(), 'su_jr_std_teouti_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    torch.save(su_jr_gaussi.cpu().detach(), 'su_jr_std_gaussi_kl'+str(int(100*beta))+'_hardtanh.pt') # Saving the tensors to plot histograms
    
    input_train_image = torch.zeros(batch_size, 1, 28, 28).cuda()
    
    input_train[:,0,0,:] = ((input_train[:,0,0,:]*train_std0)+train_mean0)
    input_train[:,0,1,:] = ((input_train[:,0,1,:]*train_std1)+train_mean1)
    input_train[:,0,2,:] = ((input_train[:,0,2,:]*train_std2)+train_mean2)
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
    fig.savefig('sumnist_train_jr_std_kl'+str(int(100 * beta))+'_hardtanh.jpg')
    
    input_test_image = torch.zeros(batch_size, 1, 28, 28).cuda()
    
    input_test[:,0,0,:] = ((input_test[:,0,0,:]*test_std0)+test_mean0)
    input_test[:,0,1,:] = ((input_test[:,0,1,:]*test_std1)+test_mean1)
    input_test[:,0,2,:] = ((input_test[:,0,2,:]*test_std2)+test_mean2)
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
    fig.savefig('sumnist_test_jr_std_kl'+str(int(100 * beta))+'_hardtanh.jpg')
    
    output_train_image = torch.zeros(batch_size, 1, 28, 28).cuda()
    
    output_train[:,0,0,:] = ((output_train[:,0,0,:]*train_std0)+train_mean0)
    output_train[:,0,1,:] = ((output_train[:,0,1,:]*train_std1)+train_mean1)
    output_train[:,0,2,:] = ((output_train[:,0,2,:]*train_std2)+train_mean2)
    output_train[:,:,:2,:] = torch.round(output_train[:,:,:2,:])
    
    print(output_train.shape)
    
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
    fig.savefig('sunet_train_jr_std_kl'+str(int(100 * beta))+'_hardtanh.jpg')
    
    output_test_image = torch.zeros(batch_size, 1, 28, 28).cuda()
    
    output_test[:,0,0,:] = ((output_test[:,0,0,:]*train_std0)+train_mean0)
    output_test[:,0,1,:] = ((output_test[:,0,1,:]*train_std1)+train_mean1)
    output_test[:,0,2,:] = ((output_test[:,0,2,:]*train_std2)+train_mean2)
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
    fig.savefig('sunet_test_jr_std_kl'+str(int(100 * beta))+'_hardtanh.jpg')
    
    # This will generate images from a vector of dimension latent_dim, where each entry is sampled from a standard gaussian (mean = 0.0 and stddev = 1.0)
    z = torch.randn(batch_size, latent_dim).cuda()
    output_gaussian = model.decode(z)
    output_gaussian[:,0,0,:] = ((output_gaussian[:,0,0,:]*train_std0)+train_mean0)
    output_gaussian[:,0,1,:] = ((output_gaussian[:,0,1,:]*train_std1)+train_mean1)
    output_gaussian[:,0,2,:] = ((output_gaussian[:,0,2,:]*train_std2)+train_mean2)
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
    fig.savefig('sunet_gauss_jr_std_kl'+str(int(100 * beta))+'_hardtanh.jpg')
    
# Ending time
end_time = time.time()

print("The total time is ",((end_time-start_time)/60.0)," minutes.")

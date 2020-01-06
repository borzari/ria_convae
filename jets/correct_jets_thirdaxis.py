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
num_particles = 100
jet_type = 'g'
learning_rate = 0.001
latent_dim = 10
beta = 0.0001 # Intensity factor
drop_prob = 0.0 # Probability to keep a node in the dropout layer
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,5), stride=(1), padding=(0))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv6 = nn.ConvTranspose2d(16, 1, kernel_size=(3,5), stride=(1), padding=(0))
        self.fc1 = nn.Linear(1 * int(num_particles - 12) * 64, 1500)
        self.fc2 = nn.Linear(1500, 2 * latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1500)
        self.fc4 = nn.Linear(1500, 1 * int(num_particles - 12) * 64)

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
        out = out.view(batch_size, 64, 1, int(num_particles - 12))
        out = self.conv4(out)
        out = torch.relu(out)
        out = self.conv5(out)
        out = torch.relu(out)
        out = self.conv6(out)
#        out[:,0,0] = 3.0 * torch.tanh(out[:,0,0])
#        out[:,0,0] = 2.5 * torch.tanh(out[:,0,0])
#        out[:,0,2] = 3.1416 * torch.tanh(out[:,0,2])
#        out[:,0,1] = torch.nn.functional.hardtanh(out[:,0,1].clone(), min_val = -3.1416, max_val = 3.1416, inplace = False)
#        out[:, :, :2] = 27 * torch.sigmoid(out[:, :, :2]) # The output x and y values should be between 0 and 27 (dimensions of a MNIST image)

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
    x_pos = torch.zeros(batch_size,3,num_particles).cuda()
    x_pos = x[:,0,:,:]

    x_pos = x_pos.view(batch_size, 3, 1, num_particles)

    # This removes the channel dimension from the tensor and makes pdist easier to use
    x_decoded_pos = torch.zeros(batch_size,3,num_particles).cuda()
    x_decoded_pos = x_decoded[:,0,:,:]

    x_decoded_pos = x_decoded_pos.view(batch_size, 3, num_particles, 1)
    x_decoded_pos = torch.repeat_interleave(x_decoded_pos, num_particles, -1)

    dist = torch.pow(pdist(x_pos, x_decoded_pos),2)

    ieo = torch.min(dist, dim = 1)

    oei = torch.min(dist, dim = 2)

    eucl = ieo.values + oei.values #+ beta * (torch.pow((x_decoded_int - get_x_int), 2) + torch.pow((x_int - get_x_decoded_int), 2))

    eucl = torch.sum(eucl) / batch_size

    reconstruction_loss = - eucl

    KL_divergence = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians

    ELBO = reconstruction_loss - KL_divergence

    loss = - ELBO

    return loss, KL_divergence, eucl

model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

eta = torch.Tensor().cuda()
phi = torch.Tensor().cuda()
pt = torch.Tensor().cuda()

etat = torch.Tensor().cuda()
phit = torch.Tensor().cuda()
ptt = torch.Tensor().cuda()

x_graph = []

tr_y_rec = []
tr_y_kl = []

tr_y_loss = []

val_y_rec = []
val_y_kl = []

val_y_loss = []

train_dataset = torch.load('all_jets_'+jet_type+'_'+str(num_particles)+'p.pt')
valid_dataset = torch.load('all_jets_val_'+jet_type+'_'+str(num_particles)+'p.pt')
#test_dataset = torch.load('MNISTsuperpixel_test_withoutzeros_onlyeight.pt')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

min_loss, stale_epochs = 999999.0, 0

for epoch in range(num_epochs):

    x_graph.append(epoch)

    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0

    val_loss_aux = 0.0
    val_kl_aux = 0.0
    val_rec_aux = 0.0

    for y, (jets_train) in enumerate(train_loader):

        if y == (len(train_loader) - 1):
            break

        input_train = jets_train.cuda()
#        input_train[:, :, :2, :] = 10 * input_train[:, :, :2, :]
#        input_train[:, :, 2, :] = beta * input_train[:, :, 2, :]

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

        if stale_epochs > 20: # if num_particles == 30
#        if stale_epochs > 20 and y > 900: # if num_particles == 100
#        if stale_epochs > 20 and y > 1050: # if num_particles == 150
            for j in range(batch_size):
                pt = torch.cat((pt,input_train[j,0,0]))
                eta = torch.cat((eta,input_train[j,0,1]))
                phi = torch.cat((phi,input_train[j,0,2]))
                ptt = torch.cat((ptt,output_train[j,0,0]))
                etat = torch.cat((etat,output_train[j,0,1]))
                phit = torch.cat((phit,output_train[j,0,2]))

    tr_y_loss.append(tr_loss_aux.item()/(len(train_loader) - 1))
    tr_y_kl.append(tr_kl_aux.item()/(len(train_loader) - 1))
    tr_y_rec.append(tr_rec_aux.item()/(len(train_loader) - 1))

    with torch.no_grad():

        for w, (jets_valid) in enumerate(valid_loader):

            if w == (len(valid_loader) - 1):
                break

            input_valid = jets_valid.cuda()

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
#    print('Epoch: {} -- Train_loss: {}'.format(epoch,tr_loss_aux.item()/(len(train_loader))))

print(len(pt),len(eta),len(phi),len(ptt),len(etat),len(phit))

int_time = time.time()
print('The time to run the network is:', (int_time - start_time)/60.0, 'minutes')

pt = pt.cpu().detach().numpy()
eta = eta.cpu().detach().numpy()
phi = phi.cpu().detach().numpy()

ptt = ptt.cpu().detach().numpy()
etat = etat.cpu().detach().numpy()
phit = phit.cpu().detach().numpy()

_, bins, _ = plt.hist(eta, bins=50, range=[-4.0, 4.0], histtype = 'step', density=False, label='Input jets eta', color = spdred)
_ = plt.hist(etat, bins=bins, histtype = 'step', density=False, label='Output jets eta', color = spdblue)
plt.xlabel('$\eta$')
plt.yscale('log')
plt.legend(loc=1)
plt.savefig('jets_'+jet_type+'_eta_thirdaxis_'+str(num_particles)+'p_fulldata_earlystopping_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')
plt.clf()

_, bins, _ = plt.hist(phi, bins=50, range=[-3.1416, 3.1416], histtype = 'step', density=False, label='Input jets phi', color = spdred)
_ = plt.hist(phit, bins=bins, histtype = 'step', density=False, label='Output jets phi', color = spdblue)
plt.xlabel('$\phi$')
plt.yscale('log')
plt.legend(loc=1) 
plt.savefig('jets_'+jet_type+'_phi_thirdaxis_'+str(num_particles)+'p_fulldata_earlystopping_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')
plt.clf()

_, bins, _ = plt.hist(pt, bins=50, range=[-50, 1500.0], histtype = 'step', density=False, label='Input jets pt', color = spdred)
_ = plt.hist(ptt, bins=bins, histtype = 'step', density=False, label='Output jets pt', color = spdblue)
plt.xlabel('$p_{T}$')
plt.yscale('log')
plt.legend(loc=1) 
plt.savefig('jets_'+jet_type+'_pt_thirdaxis_'+str(num_particles)+'p_fulldata_earlystopping_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')
plt.clf()

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
plt.savefig('loss_jets_'+jet_type+'_thirdaxis_'+str(num_particles)+'p_fulldata_earlystopping_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.png')

# Save the model
torch.save(model.state_dict(), 'model_jets_'+jet_type+'_thirdaxis_'+str(num_particles)+'p_fulldata_earlystopping_latentdim'+str(latent_dim)+'_'+str(num_epochs)+'epochs.pt')

sum = 0

#for y, (images_test, labels_test) in enumerate(test_loader):

#    if y == (len(test_loader) - 1):
#        break

#    input_test = images_test[:, :, :].cuda()
#    input_test[:, :, 2, :] = 27 * input_test[:, :, 2, :]

#    output_test = model(input_test)
#    loss, kl, eucl = compute_loss(model, input_test) # kl and eucl are not being used here

#    sum += loss.item()

#print('The average test loss is: ',sum/(len(test_loader) - 1))

# Ending time
end_time = time.time()

print("The total time is ",((end_time-start_time)/60.0)," minutes.")

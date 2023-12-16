# Install FrEIA amd torch via pip install

from time import time
import torch
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from FrEIA.framework import InputNode, OutputNode, ConditionNode, Node, GraphINN
from FrEIA.modules import AllInOneBlock
from StandardScaler import StandardScaler as stdScaler

############################################
# Initialize Parameters
############################################

n_in = 2
n_cond = 2
n_CB = 10
n_subnet = 20
file_in = 'in_test.txt'
file_out = 'out_test.txt'


############################################
# Function for cINN
############################################


# Subnet Generator for cINN coupling blocks
def genSubnet(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, n_subnet), nn.BatchNorm1d(n_subnet), nn.LeakyReLU(),
                         nn.Linear(n_subnet, c_out))


# Define loss method, max likelyhood
def max_Likelyhood(z, log_jac_det):
    return torch.mean(0.5 * torch.sum(z ** 2, dim=1) - log_jac_det)


# Define accuracy meassurement R2
def R2(inputs, targets):
    R2 = [np.corrcoef(inputs[:, i].cpu().detach().numpy(), targets[:, i].cpu().detach().numpy())[
              0, 1] ** 2 for i in range(inputs.shape[1])]
    return np.array(R2)


# Define accuracy meassurement RMSE
def RMSE(inputs, targets):
    inp = inputs.cpu().detach().numpy()
    tar = targets.cpu().detach().numpy()
    return np.sqrt(np.sum((inp - tar) ** 2)/len(inp))


############################################
# create cINN
############################################

# 1. Conditional Node
cond = ConditionNode(n_cond, name='condition')

# 2. List with Nodes starting with input node
nodes = [InputNode(n_in, name='input')]

# 3. Chain demanded number of coupling blocks to node list
for i in range(n_CB):
    nodes.append(Node(nodes[-1],
                      AllInOneBlock,
                      {'subnet_constructor': genSubnet, 'permute_soft': True},
                      conditions=cond,
                      name=F'coupling_{i}'))

# 4. finish with output node
nodes.append(OutputNode([nodes[-1].out0], name='Output'))

# 5. Create cINN from node lists
cINN = GraphINN(nodes + [cond], verbose=True)

############################################
# Prepare Training
############################################

# Setting learning parameters
learning_rate = 0.1
batch_size = 25
epochs = 200
# collect trainable parameters
trainable_parameters = [p for p in cINN.parameters() if p.requires_grad]
# define optimizer
optimizer = torch.optim.Adagrad(trainable_parameters, lr=learning_rate)
# Initialize trainable parameters
for param in [p for p in trainable_parameters]:
    param.data = 0.025 * torch.randn_like(param)
# Initialize learining rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

############################################
# Prepare Training
############################################

# Load data
In_train = np.loadtxt(file_in, delimiter=';', skiprows=1)
Out_train = np.loadtxt(file_out, delimiter=';', skiprows=1)

# Initialize Standard Scaler
in_scaler = stdScaler()
out_scaler = stdScaler()

# Transform data
In_train = in_scaler.transform(In_train, set_par=True)
Out_train = out_scaler.transform(Out_train, set_par=True)

# Convert to torch tensors
In = torch.tensor(In_train, dtype=torch.float, device='cpu')
Out = torch.tensor(Out_train, dtype=torch.float, device='cpu')

# Set division for training samples and validation samples
n_train = int(np.round(In.shape[0] * 0.75))
cINN.to('cpu')

# Divide Data
In_val = In[n_train:, :]
Out_val = Out[n_train:, :]

# Create a data loader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(In[:n_train, :], Out[:n_train, :]),
    batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(In[n_train:, :], Out[n_train:, :]),
    batch_size=batch_size, shuffle=True, drop_last=True)

############################################
# Perform training
############################################

# Initialize loss and accuracy arrays
train_loss = np.zeros([epochs])
val_loss = np.zeros([epochs])
val_acc = np.zeros([epochs])

# initialize output
print('\n\n| Epoch:    |  Time:  | Learning rate:   | Loss_maxL_train: |  | Acc_i_val: | L_maxL_val: |')
print('|-----------|---------|------------------|------------------|  |------------|-------------|')
t_start = time()

# Start Training
for i_epoch in range(epochs):
    t_epoch = time()
    # set to train mode, it is necessary for certain torch layers, like batch normalization
    cINN.train()

    # set epcoh loss_history empty
    loss_history = []

    ###############################################
    # start training over data
    for x, y in train_loader:
        optimizer.zero_grad()

        # forward step
        z, log_jac_det = cINN.forward(x, c=y, rev=False, jac=True)
        # maxlikelyhood loss:
        l_maxL = max_Likelyhood(z, log_jac_det)
        # Backpropagation:
        l_maxL.backward()
        # save loss
        loss_history.append(l_maxL.item())
        # step the optimizer
        optimizer.step()

    train_loss[i_epoch] = np.mean(np.array(loss_history), axis=0)

    ###############################################
    # set cINN to eval mode, necessary for layers such as batch normalization
    cINN.eval()
    # eval without performaing any gradient descent
    with torch.no_grad():
        # set epcoh loss_history and acc_history to empty
        loss_history = []
        acc_history = []
        # start validation over data
        for x, y in val_loader:
            # forward step
            z, log_jac_det = cINN.forward(x, c=y, rev=False, jac=True)
            # maxlikelyhood loss:
            l_maxL = max_Likelyhood(z, log_jac_det)
            # inverse: sample z from normal distribution and inverse cINN
            z = torch.randn_like(z, dtype=torch.float, device='cpu')
            x_rec, log_jac_det = cINN.forward(z, c=y, rev=True, jac=False)

            # calculate accuracy R2 of postrior prediction to true value
            acc_history.append(RMSE(x, x_rec))
            # save loss
            loss_history.append(l_maxL.item())

        val_loss[i_epoch] = np.mean(np.array(loss_history), axis=0)
        val_acc[i_epoch] = np.mean(np.stack(acc_history, axis=0), axis=0)

    lr_scheduler.step()

    # Print results
    print('| %4d/%4d | %6ds | %16.10f | %16.3f |  | %10.3f | %11.3f |' % (
        i_epoch + 1, epochs, min((time() - t_epoch), 99999999), lr_scheduler.get_last_lr()[0],
        min(train_loss[i_epoch], 9999999),
        min(val_acc[i_epoch], 999999),
        min(val_loss[i_epoch], 9999999)))

# Print final lines
print('|-----------|---------|------------------|------------------|  |------------|-------------|')
print(f"\n\nTraining took {(time() - t_start) / 60:.2f} minutes\n")

############################################
# Validate
############################################

# Latent sample size
lat_sample_size = 1000

# define input and output for validation
i_val = In_val
o_val = Out_val

# Perform predictions based on the output data
# Data is already scaled with the scaler

cINN.eval()
with torch.no_grad():

    # Create random latent space
    rnd_lat = torch.randn(lat_sample_size * In_val.shape[0], n_in, dtype=torch.float, device='cpu')

    # reshape output data for conditions alligning with the latent space size
    # o_val = o_val.repeat([lat_sample_size,1])
    o_val = torch.repeat_interleave(o_val, lat_sample_size, dim=0)

    # Perform prediction with cINN
    inputs, jac = cINN.forward(rnd_lat, c=o_val, rev=True, jac=False)

    # Calculate mean values for each input value by averaging over the latent sample size
    in_mean = np.zeros_like(In_val.numpy())

    for i in range(in_mean.shape[0]):
        in_mean[i, :] = torch.mean(inputs[i * 1000:(i + 1) * 1000, :], 0).cpu().detach().numpy()

#######################################################
# Visualize results
fig, axs = plt.subplots(1, n_in, figsize=(15, 5))  # 1 row, 3 columns

# scatter plot of prediction vs. real value for a number of samples.
for i in range(n_in):
    axs[i].scatter(i_val[:, i], in_mean[:, i])
    axs[i].set_ylim([-3, 3])
    axs[i].set_xlim([-3, 3])
fig.savefig('Multiple_samples.png', dpi=300)
fig.close()

# histogram for a single sample
fig, axs = plt.subplots(1, n_in, figsize=(15, 5))
for i in range(n_in):
    axs[i].hist(inputs[:1000, i].detach().numpy(), bins=50, alpha=0.5, color='blue', edgecolor='black')
    axs[i].hist(rnd_lat[:1000, i].detach().numpy(), bins=50, alpha=0.5, color='green', edgecolor='black')
    axs[i].set_xlim([-3, 3])
fig.savefig('Sample_Histogram.png', dpi=300)
fig.close()

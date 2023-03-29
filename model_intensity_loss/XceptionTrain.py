# !/usr/bin/env python
# -*- encoding: utf-8 -*- #

# ----------------------------------------------------------------------------------------------------------
# File Name:        XceptionTrain.py
# Author:           Renxi Liu, Xianyuer, Erjie Wu
# Version:          1.1
# Created:          2023/03/22 16:10:58
# Description:      Main Function: Generate data in real time and training
#                   Cross Reference: Zernike, fun, propagation_speed, Xception_model
# Function List:    None for outer use
# Input List:
#         <name>       <type>        <description>
#         INPUT        .json         All input parameters
# Output List:
#         <name>       <type>        <description>
#         $model_name  .log          History of loss
#         $model_name  .pt           Model from machine learning
# History: 
#         <author>     <version>     <time>			<description>
#         Xianyuer     1.0           Unknown        Creat the file
#         Erjie Wu     1.1           2023/03/22		Optimize the structure & Add some notes
# ----------------------------------------------------------------------------------------------------------


#------------ Load Package ------------#

####### Package: Commonly used #######
import numpy as np
import time
import json
import random
import sys
sys.path.append("..")
import matplotlib.pyplot as plt

####### Package: Scikit-Learn #######
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,MinMaxScaler, StandardScaler

####### Package: Pytorch #######
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import Dataset,DataLoader
from torchinfo import summary

####### Package: User-defined #######
from Zernike import *
from fun import *
from propagation_speed import *
from Xception_model import *


#------------ Load & Set Parameters ------------#

####### Load Input File #######
with open("INPUT.json", 'r', encoding='utf-8') as fw:
    injson = json.load(fw)

####### Load Parameters for Pysical Model #######
mm = injson['Train']['PM']['OrderOfGrid'] # Log2 of the number of grid size
mgs = injson['Train']['PM']['TruncationIndex'] # Truncation index of the beam
a0 = injson['Train']['PM']['RadiusOfSource'] # Initial size of the beam
xx0 = injson['Train']['PM']['BufferingMultipleOfSource'] # Buffering multiple of source plane
plm = injson['Train']['PM']['WaveLength'] # wave length of the beam
zfh = injson['Train']['PM']['TransmissionDistance'] # Transmission distance
xxz = injson['Train']['PM']['BufferingMultipleOfFocus'] # Buffering multiple of focus plane
minZnkDim = injson['Train']['PM']['minZnkDim'] # Minimum dimension of the Zernike Polynomial
maxZnkOrder = injson['Train']['PM']['maxZnkOrder'] # Maximum Order of the Zernike Polynomial
rms = injson['Train']['PM']['rms'] # Root mean square(RMS) of the phase 
eeznk = injson['Train']['PM']['ZnkDecay'] # Decay index of coefficients of Zernike Polynomial
zernike_dir = injson['Train']['PM']['dir'] # Directory

####### Load Parameters for Machine Learning #######
model_path = injson['Train']['ML']['InputModelDir'] # Directory of model used for initialization
input_model = injson['Train']['ML']['InputModelOption'] # Option for whether use other model for initialization or not
epoch = injson['Train']['ML']['Epoch'] # Number of epochs for training
batch_size = injson['Train']['ML']['Batch_Size'] # Batch size for training
seed = injson['Train']['ML']['RandomNumberSeed'] # Random number seed
lr = injson['Train']['ML']['InitialLearningRate'] # Initial learning rate
lr_gamma = injson['Train']['ML']['LearningRateDecay'] # Decay index for learning rate
lr_step = injson['Train']['ML']['LearningRateDecay_Step'] # Number of epochs trained after each attenuation
mgs_guass = injson['Train']['ML']['GaussTruncationIndex'] # Truncation index for gaussian musk
lossform = injson['Train']['ML']['LossForm'] # Form of loss function, "Zernike" or "Intensity"

####### Load & Set Running Parameters #######
# Model Naming Convention: Date_ZernikeOrder_GridSize_Epochs_BatchSize_rms_LearningRate(3to8:1e-3~1e-8)
model_name = "%s_%d_%d_%d_%d_%d_%dto%d"%(time.strftime('%m%d'), maxZernike(maxZnkOrder), pow(2,mm), epoch, batch_size, rms, int(-np.log10(lr)), int(epoch/lr_step-np.log10(lr))) 

dir = injson['Train']['Dir'] # Directory for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Device for training

####### Calculate Other Parameters Needed #######
# Coefficient used for inversion of Zernike Polynomial with even order
Zernike_aliaslist = []
for zi in range(3, maxZnkOrder+2):
    Zernike_aliaslist = Zernike_aliaslist + [pow(-1,zi)]*zi
Zernike_alias = np.array(Zernike_aliaslist, dtype=np.float32) 
Zernike_alias  = torch.tensor(Zernike_alias).to(device)

# Calculate the value of the Zernike function on the grid
Zer = Zer1(maxZnkOrder, mm, a0, xx0) 
Zer = torch.tensor(Zer).to(device)

# Calculate the amplitude distribution of the initial beam
init_intens = init_intensity(mm, a0, xx0, mgs) 
init_intens = torch.tensor(init_intens).to(device)

# Calculate some middle parameters
ngrid = pow(2,mm) # Grid number
n1 = ngrid/2 + 1
aa0 = xx0*a0 # Initial beam radius after buffering
dxy0 = aa0/ngrid # The actual length of the unit grid at the source plane after buffering
airy = 1.22*plm*zfh/(2*a0) # Radius of real Airy disk at the focus plane
aaz = airy*xxz # Radius of Airy disk after buffering
dxyz = aaz/ngrid # The actual length of the unit grid at the focus plane after buffering
ngrid2 = ngrid//2
a02 = a0*a0
dlta = (1-aaz/aa0)/zfh # delta, used in Adaptive Coordinate Transformation(ACT)
ddxz = 1-dlta*zfh # D(z=z_f), used in ACT
zzzz = zfh/ddxz # z in new coordinate
wave_number = 2*torch.pi/plm # wave number

# Calculate the discretized coordinate
gy,gx = np.meshgrid(dxy0*np.linspace(1-n1, ngrid-n1, ngrid), dxy0*np.linspace(1-n1, ngrid-n1, ngrid)) # Grid at source plane
gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1, ngrid-n1, ngrid), dxyz*np.linspace(1-n1, ngrid-n1, ngrid)) # Grid at focus plane

# Calculate the coefficient correspondence in the inverse space
h = torch.zeros(ngrid).to(device)
h_sum = torch.zeros((ngrid,ngrid)).to(device)
prop1(ngrid, n1, zzzz, wave_number, aa0, h)
for i in range(ngrid):
    for j in range(ngrid):
        h_sum[i,j] = h[i]+h[j]
        
# Calculate the phase
ei = -wave_number*(gx*gx+gy*gy)/2*(1/zfh) # Phase of initial field
ei = torch.tensor(ei).to(device)
ec = wave_number*gx*gx*dlta/2 + wave_number*gy*gy*dlta/2
ec = torch.tensor(ec).to(device)
ez = -1*wave_number*gx2*gx2*dlta/(2*ddxz) - wave_number*gy2*gy2*dlta/(2*ddxz) # phi(x,y,z), used in ACT
ez = torch.tensor(ez).to(device)
f_m = torch.exp(1j*ei)*torch.exp(1j*ec)

# Calculate mask for initial beam
mask0 = ((gx**2 + gy**2)/a02 <= 1) 
mask0 = torch.tensor(mask0).to(device)

# Calculate the Gaussian Matrix
r02 = a0*a0*xx0*xx0/9
gauss = np.zeros((ngrid,ngrid))
mask00 = ((gx*gx + gy*gy)<=r02)
a = np.exp(-1*pow(((gx*gx+gy*gy)/a02), mgs_guass)) 
gauss = a * mask00
gauss = torch.tensor(gauss).to(device)


#------------ Training Process ------------#

fid = open(model_name+'.log', 'w') # Record the training history

def fit(net, batch_size, epochs, learning_rate, Zernike_alias, ngrid, ngrid2, init_intens, Zer, mask0, f_m, h_sum, ez, ddxz, Lr_gamma, Lr_step, gauss):
    # Model Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate) 
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = Lr_step, gamma = Lr_gamma)
    Zernike_alias_0 = torch.zeros((batch_size, maxZernike(maxZnkOrder)-2)).to(device)  
    for i in range(batch_size):
        Zernike_alias_0[i,:] = Zernike_alias
    # Loss function
    criterion=torch.nn.MSELoss(reduction='mean')  
    # Progress monitor
    samples = 0
    # Accuracy monitor
    corrects = 0

    # Train for some epochs
    for epoch in range(epochs):
    # Train for each batch
        # Generate data
        y = cc(batch_size, maxZnkOrder, "random", eeznk, rms, zernike_dir)
        y = y[:,2:]
        y = torch.tensor(y).to(device)
        x = nor_progagtion(batch_size, ngrid, ngrid2, init_intens, y, Zer, mask0, f_m, h_sum, ez, ddxz)
        x = torch.reshape(x, [batch_size, 1, pow(2,mm), pow(2,mm)]).to(device)  
        # Forward propagation
        cz_pred = net(x)

        # Calculate the loss
        if lossform == "Intensity":
            # # change loss
            # loss_z1 = torch.sum(pow(cz_pred - y,2))
            # loss_z2 = torch.sum(pow(cz_pred*Zernike_alias - y,2))
            # if loss_z1>loss_z2:
            #     cz_pred = cz_pred * Zernike_alias
            far_field_intens_pred = nor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,mask0,f_m,h_sum,ez,ddxz)
            far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, pow(2,mm), pow(2,mm)]).to(device)  
            loss = torch.mean(pow((far_field_intens_pred - x)*gauss,2))
        elif lossform == "Zernike":
            loss = criterion(cz_pred.float(),y.float())

        # Clear the gradient
        opt.zero_grad()
        # Backward propagation
        loss.backward()
        # Update the gradient
        opt.step()
        
        # Monitor progress: With each batch trained, the data seen by the model increases x.shape[0].
        samples += x.shape[0]
        # print the process after every 100 epochs & at the end
        if (epoch + 1) % (100) == 0 or epoch  == (epochs - 1):
            loss_z1 = torch.mean(pow(cz_pred - y,2))
            loss_z2 = torch.mean(pow(cz_pred*Zernike_alias_0 - y,2))
            loss_zernike = np.minimum(loss_z1.detach().cpu().numpy(), loss_z2.detach().cpu().numpy()) 
            # loss_zernike = np.mean(pow(cz_pred.detach().cpu().numpy() - y.detach().cpu().numpy(),2))
            print("Epoch{}:[{}/{} {: .0f}%], Loss:{:.6f} ".format(
                epoch + 1
                , samples
                , epochs*batch_size
                , 100*samples/(epochs*batch_size)
                , loss.data.item()), "Zer loss: ", loss_zernike)
            if lossform == "Intensity":
                fid.write(str(loss.cpu().item())+'\t'+str(loss_zernike))
            elif lossform == "Zernike":
                fid.write(str(loss.cpu().item()))
            fid.write('\n')

    # Model save
    torch.save(net.state_dict(), './model/'+model_name+'.pt')

# Set the random number seed
torch.manual_seed(51)

# Instantiate the model
net = Xception()
if input_model == 1:
    net.load_state_dict(torch.load(model_path))
net = net.to(device)

fit(net, batch_size, epoch, lr, Zernike_alias, ngrid, ngrid2, init_intens, Zer, mask0, f_m, h_sum, ez, ddxz, lr_gamma, lr_step, gauss)

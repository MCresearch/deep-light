# !/usr/bin/env python
# -*- encoding: utf-8 -*- #
 
# ----------------------------------------------------------------------------------------------------------
# File Name:        ModelTest.py
# Author:           Xianyuer, Erjie Wu
# Version:          1.1
# Created:          2023/03/22 19:42:56
# Description:      Main Function: Test the performance of given model
#                   Cross Reference: Zernike, fun, propagation_speed, Xception_model
# Function List:    None for outer use
# Input List:
#         <name>       <type>        <description>
#         INPUT        .json         All input parameters
# Output List:
#         <name>       <type>        <description>
#         rms          .log          Root mean square of phase for each frame
#         *_int        .png          Figure of intensity distribution for real, predict and residual cases
#         *_zer        .png          Comparison of Zernike coefficient between reality and prediction 
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
import os

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
mm = injson['Test']['PM']['OrderOfGrid'] # Log2 of the number of grid size
mgs = injson['Test']['PM']['TruncationIndex'] # Truncation index of the beam
a0 = injson['Test']['PM']['RadiusOfSource'] # Initial size of the beam
xx0 = injson['Test']['PM']['BufferingMultipleOfSource'] # Buffering multiple of source plane
plm = injson['Test']['PM']['WaveLength'] # wave length of the beam
zfh = injson['Test']['PM']['TransmissionDistance'] # Transmission distance
xxz = injson['Test']['PM']['BufferingMultipleOfFocus'] # Buffering multiple of focus plane
minZnkDim = injson['Test']['PM']['minZnkDim'] # Minimum dimension of the Zernike Polynomial
maxZnkOrder = injson['Test']['PM']['maxZnkOrder'] # Maximum Order of the Zernike Polynomial
rms = injson['Test']['PM']['rms'] # Root mean square(RMS) of the phase 
eeznk = injson['Test']['PM']['ZnkDecay'] # Decay index of coefficients of Zernike Polynomial
zernike_dir = injson['Test']['PM']['dir'] # Directory of Zernike coefficient for confirm case

####### Load Parameters for Machine Learning #######
model_name = injson['Test']['ML']['ModelName'] # Name of tested model
model_path = injson['Test']['ML']['InputModelDir'] # Directory of model used for initialization
input_model = injson['Test']['ML']['InputModelOption'] # Option for whether use other model for initialization or not
batch_size = injson['Test']['ML']['Batch_Size'] # Batch size for Testing
seed = injson['Test']['ML']['RandomNumberSeed'] # Random number seed
snapshoot = injson['Test']['ML']['FrameNumber'] # Frame number for testing

####### Load & Set Running Parameters #######
if os.path.exists(model_name):
    print("文件已经存在")
else:
    os.mkdir(model_name)
dir = "./"+model_name+"/" # Directory for saving testing result
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Device for testing
print(dir)

####### Calculate Other Parameters Needed #######
# Coefficient used for inversion of Zernike Polynomial with even order
Zernike_aliaslist = []
for zi in range(3, maxZnkOrder+2):
    Zernike_aliaslist = Zernike_aliaslist + [pow(-1,zi)]*zi
Zernike_alias = np.array(Zernike_aliaslist, dtype=np.float32) 
Zernike_alias  = torch.tensor(Zernike_alias).to(device)

# Calculate the value of the Zernike function on the grid
Zer = Zer1(maxZnkOrder,mm,a0,xx0)
Zer = torch.tensor(Zer).to(device)

# Calculate the amplitude distribution of the initial beam
init_intens = init_intensity(mm,a0,xx0,mgs)
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
nzer = maxZernike(maxZnkOrder)-2 # Zernike order used in model

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


#------------ Testing Process ------------#

# Instantiate the model
net = Xception()
net.load_state_dict(torch.load(model_path))
net = net.to(device)

# Parameter Initialization
loss_int= torch.zeros([snapshoot,1],dtype=torch.float).to(device)  
loss_zer= torch.zeros([snapshoot,nzer],dtype=torch.float).to(device)
y_predict_choose = torch.zeros([snapshoot,nzer],dtype=torch.float).to(device)
rms_save = torch.zeros([snapshoot,2],dtype=torch.float).to(device)
x_diff = torch.zeros([snapshoot, ngrid, ngrid],dtype=torch.float).to(device)

# Generate samples
y_test = cc(snapshoot,maxZnkOrder,"random",eeznk,rms,zernike_dir)
y_test = y_test[:,2:]
y_test = torch.tensor(y_test).to(device)
        
for i in range(snapshoot):

    c_test = torch.reshape(y_test[i,:], [1, nzer]).to(device)  

    # Get the real intensity distribution
    x_test = nor_progagtion(batch_size,ngrid,ngrid2,init_intens,c_test,Zer,mask0,f_m,h_sum,ez,ddxz)
    x_test = torch.reshape(x_test, [batch_size, 1, ngrid, ngrid]).to(device)  

    # Get the predict Zernike coefficients and intensity distribution
    y_predict = net(x_test.reshape([batch_size,1,ngrid,ngrid]))
    x_predict = nor_progagtion(batch_size,ngrid,ngrid2,init_intens,y_predict,Zer,mask0,f_m,h_sum,ez,ddxz)
    
    # Choose the reasonable loss
    loss_z1 = torch.mean(pow(y_predict[0,:] - c_test[0,:],2))
    loss_z2 = torch.mean(pow(y_predict[0,:]*Zernike_alias - c_test[0,:],2))
    if loss_z1>loss_z2:
        y_predict_choose[i,:] = y_predict[0,:] * Zernike_alias
    else:
        y_predict_choose[i,:] = y_predict[0,:]

    # Performance measurement   
    loss_int[i]= torch.mean(pow((x_predict - x_test[:,0,:,:]),2))  
    rms_save[i,0] = torch.sum(pow(c_test-y_predict[0,:],2))
    rms_save[i,1] = torch.sum(pow(c_test-y_predict_choose[i,:],2))

    # Calculate the residual far-field intensity distribution
    x_diff =  nor_progagtion(batch_size,ngrid,ngrid2,init_intens,c_test-y_predict_choose[i,:],Zer,mask0,f_m,h_sum,ez,ddxz)
    if rms_save[i,1] > 2:
        print("bad predict:rms=",rms_save[i,1])
        plt.figure(1, figsize=(12,4))
        plt.subplot(121)
        plt.bar(np.array(range(nzer)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
        plt.bar(np.array(range(nzer)),y_predict.detach().cpu().numpy()[0,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
        plt.xlabel("Zernike order",fontsize=15)
        plt.ylabel("Zernike coefficient values",fontsize=15)
        plt.xticks(size = 10)
        plt.yticks(size=10)
        # plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
        plt.legend()

        plt.subplot(122)
        plt.bar(np.array(range(nzer)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
        plt.bar(np.array(range(nzer)),y_predict_choose.detach().cpu().numpy()[i,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
        plt.xlabel("Zernike order",fontsize=15)
        plt.ylabel("Zernike coefficient values",fontsize=15)
        plt.xticks(size = 10)
        plt.yticks(size=10)
        # plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
        plt.legend()
        plt.savefig(dir+str(i+1)+"_zer.png",bbox_inches='tight') #,bbox_inches='tight'
        plt.close()

        plt.figure(1, figsize=(16,4))
        plt.subplot(131)
        plt.contourf(x_test.detach().cpu().numpy()[0,0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
        plt.colorbar()
        # plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
        # plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
        plt.xlabel("x (m)",fontsize=15)
        plt.ylabel("y (m)",fontsize=15)

        plt.subplot(132)
        plt.contourf(x_predict.detach().cpu().numpy()[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
        plt.colorbar()
        # plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
        # plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
        plt.xlabel("x (m)",fontsize=15)
        plt.ylabel("y (m)",fontsize=15)


        plt.subplot(133)
        plt.contourf(x_diff.detach().cpu().numpy()[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
        plt.colorbar()
        # plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
        # plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
        plt.xlabel("x (m)",fontsize=15)
        plt.ylabel("y (m)",fontsize=15)

        plt.savefig(dir+str(i+1)+"_int.png",bbox_inches='tight')
        plt.close()

fid = open(dir+'rms.log', 'w')
rms_mean = torch.mean(rms_save[i,1])
fid.write(str(rms_mean.item())+'\n')
for i in range(snapshoot):
    fid.write(str(rms_save[i,0].item())+'\t'+str(rms_save[i,1].item())+'\n')

plt.figure(1, figsize=(12,4))
plt.subplot(121)
plt.bar(np.array(range(nzer)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
plt.bar(np.array(range(nzer)),y_predict.detach().cpu().numpy()[0,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
plt.xlabel("Zernike order",fontsize=15)
plt.ylabel("Zernike coefficient values",fontsize=15)
plt.xticks(size = 10)
plt.yticks(size=10)
# plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
plt.legend()

plt.subplot(122)
plt.bar(np.array(range(nzer)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
plt.bar(np.array(range(nzer)),y_predict_choose.detach().cpu().numpy()[snapshoot-1,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
plt.xlabel("Zernike order",fontsize=15)
plt.ylabel("Zernike coefficient values",fontsize=15)
plt.xticks(size = 10)
plt.yticks(size=10)
# plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
plt.legend()
plt.savefig(dir+"_zer.png",bbox_inches='tight') #,bbox_inches='tight'
plt.close()

plt.figure(1, figsize=(16,4))
plt.subplot(131)
plt.contourf(x_test.detach().cpu().numpy()[0,0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
plt.colorbar()
# plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
# plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)

plt.subplot(132)
plt.contourf(x_predict.detach().cpu().numpy()[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
plt.colorbar()
# plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
# plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)



plt.subplot(133)
plt.contourf(x_diff.detach().cpu().numpy()[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
plt.colorbar()
# plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
# plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)

plt.savefig(dir+"_int.png",bbox_inches='tight')
plt.close()
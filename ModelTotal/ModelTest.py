# !/usr/bin/env python
# -*- encoding: utf-8 -*- #
 
# ----------------------------------------------------------------------------------------------------------
# File Name:        ModelTest.py
# Author:           Xianyuer, Erjie Wu
# Version:          1.1
# Created:          2023/03/22 19:42:56
# Description:      Main Function: Test the performance of given model
#                   Cross Reference: Zernike, fun, propagation_speed, Xception_model, TestPlot
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
#         Erjie Wu     1.2           2023/03/27     Add rms-test and SR-test function
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
from TestPlot import *


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

####### Load Parameters for Plot #######
basic_op = injson['Test']['PL']['Basic'] # Whether show the basic result
badplot = injson['Test']['PL']['BadPlot'] # Whether show the bad result
rms_op = injson['Test']['PL']['RMSStatistics'] # Whether show the rms result
SR_op = injson['Test']['PL']['SRStatistics'] # Whether show the SR result
histbin = injson['Test']['PL']['BinsOfHist'] # Set the bins of hist
bias_op = injson['Test']['PL']['Bias'] # Whether show the bias result
bias_pos = injson['Test']['PL']['BiasForm'] # The zernike order where bias is added
bias_int = injson['Test']['PL']['BiasIntensity'] # The intensity of bias
bias_st = injson['Test']['PL']['BiasStatistics'] # Whether show the bias statistics

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
loss_int = torch.zeros([snapshoot,1],dtype=torch.float).to(device)  
loss_zer = torch.zeros([snapshoot,nzer],dtype=torch.float).to(device)
Sr = np.zeros([snapshoot,1])
y_predict_choose = torch.zeros([snapshoot,nzer],dtype=torch.float).to(device)
rms_save = torch.zeros([snapshoot,2],dtype=torch.float).to(device)
rms_biasit = torch.zeros(snapshoot,dtype=torch.float).to(device)
rms_res = torch.zeros(snapshoot,dtype=torch.float).to(device)
x_diff = torch.zeros([snapshoot, ngrid, ngrid],dtype=torch.float).to(device)

# Generate samples
y_test = cc(snapshoot,maxZnkOrder,"random",eeznk,rms,zernike_dir)
# Bias added
if bias_op == 1:
    y_test[:,bias_pos-1] = y_test[:,bias_pos-1] + bias_int
    rms_bias = np.zeros(snapshoot)
    for bi in range(snapshoot):
        rms_bias[bi] = np.sum(pow(y_test[bi,:],2))
y_test = y_test[:,2:]
y_test = torch.tensor(y_test).to(device)

# Unperturbed situation
y_zero = cc(snapshoot,maxZnkOrder,"zero",eeznk,rms,zernike_dir)
y_zero = torch.tensor(y_zero[:,2:]).to(device)
        
for i in range(snapshoot):

    c_test = torch.reshape(y_test[i,:], [1, nzer]).to(device) 
    c_zero = torch.reshape(y_zero[i,:], [1, nzer]).to(device) 

    # Get the real&zero intensity distribution
    x_test = nor_progagtion(batch_size,ngrid,ngrid2,init_intens,c_test,Zer,mask0,f_m,h_sum,ez,ddxz)
    x_test = torch.reshape(x_test, [batch_size, 1, ngrid, ngrid]).to(device)  
    x_zero = progagtion(batch_size,ngrid,ngrid2,init_intens,c_zero,Zer,mask0,f_m,h_sum,ez,ddxz)

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
    rms_biasit[i] = pow(y_test[i,bias_pos-3]-y_predict_choose[i,bias_pos-3],2)
    rms_res[i] = rms_save[i,1]-rms_biasit[i]

    # Calculate the residual far-field intensity distribution & SR
    x_diff =  nor_progagtion(batch_size,ngrid,ngrid2,init_intens,c_test-y_predict_choose[i,:],Zer,mask0,f_m,h_sum,ez,ddxz)
    x_diff0 = progagtion(batch_size,ngrid,ngrid2,init_intens,c_test-y_predict_choose[i,:],Zer,mask0,f_m,h_sum,ez,ddxz)
    Sr[i] = torch.max(torch.max(x_diff0[0,:,:])).detach().cpu().numpy()/torch.max(torch.max(x_zero[0,:,:])).detach().cpu().numpy()
    
    # Visualization for bad situations
    if rms_save[i,1] > 2 and badplot == 1:
        print("bad predict:rms=",rms_save[i,1])

        # Plot the Zernike coefficient comparison
        plt.figure(1, figsize=(12,4))
        plt.subplot(121)
        ZerCom(np.array(range(nzer)),c_test[0,:],y_predict[0,:])
        plt.subplot(122)
        ZerCom(np.array(range(nzer)),c_test[0,:],y_predict_choose[i,:])
        plt.savefig(dir+str(i+1)+"_zer.png",bbox_inches='tight') 
        plt.close()

        # Plot the intensity distribution camparison
        IntCom(x_test[0,0,:,:],x_predict[0,:,:],x_diff[0,:,:],name=dir+str(i+1)+"_int.png")
        
# Basic result: Zernike coefficient and intensity distribution output
if basic_op == 1:
    # Plot the Zernike coefficient comparison
    plt.figure(1, figsize=(12,4))
    plt.subplot(121)
    ZerCom(np.array(range(nzer)),c_test[0,:],y_predict[0,:])
    plt.subplot(122)
    ZerCom(np.array(range(nzer)),c_test[0,:],y_predict_choose[snapshoot-1,:])
    plt.savefig(dir+"rms=%d_zer.png"%(rms),bbox_inches='tight') 
    plt.close()

    # Plot the intensity distribution camparison
    IntCom(x_test[0,0,:,:],x_predict[0,:,:],x_diff[0,:,:],name=dir+"rms=%d_int.png"%(rms))

# RMS analysis
if rms_op == 1:
    # Output the prediction rms
    fid = open(dir+'rms_rms=%d.log'%(rms), 'w')
    rms_mean = torch.mean(rms_save[:,1])
    fid.write(str(rms_mean.item())+'\n')
    for i in range(snapshoot):
        fid.write(str(rms_save[i,0].item())+'\t'+str(rms_save[i,1].item())+'\n')
    fid.close()

    # Plot the RMS statistical result
    plt.figure(1, figsize=(6,4))
    plt.hist(rms_save[:,1].detach().cpu().numpy(), bins=histbin)
    plt.xlabel("rms")
    plt.ylabel("number")
    plt.title("Prediction RMS distribution for Phase Varience %d"%(rms))
    plt.grid(True)
    plt.savefig(dir+"rms=%d.png"%(rms),bbox_inches='tight')
    plt.close()

# Plot the SR statistical result
if SR_op == 1:
    plt.figure(1, figsize=(6,4))
    plt.hist(Sr, bins=histbin)
    plt.xlabel("SR")
    plt.ylabel("number")
    plt.title("Prediction SR for Phase Varience %d"%(rms))
    plt.grid(True)
    plt.savefig(dir+"rms=%d_SR.png"%(rms),bbox_inches='tight')
    plt.close()

# Plot the analysis of bias
if bias_op == 1:
    plt.figure(1, figsize=(12,4))
    plt.subplot(121)
    plt.scatter(rms_bias, rms_save[:,1].detach().cpu().numpy())
    plt.xlabel("Phase varience")
    plt.ylabel("RMS")
    plt.title("RMS analysis of intensity %.2f for Phase varience=%d"%(bias_int,rms))

    plt.subplot(122)
    plt.scatter(rms_bias, Sr)
    plt.xlabel("Phase varience")
    plt.ylabel("SR")
    plt.title("SR analysis of intensity %.2f for Phase varience=%d"%(bias_int,rms))

    plt.savefig(dir+"rms=%d_bias=%.2f.png"%(rms,bias_int),bbox_inches='tight')
    plt.close()

    fid = open(dir+'bias_rms=%d.log'%(rms), 'a')
    fid.write(str(bias_int)+'\t'+str(torch.sqrt(torch.mean(rms_save[:,1])).item())+'\t'+str(torch.sqrt(torch.mean(rms_biasit)).item())+'\t'+str(torch.sqrt(torch.mean(rms_res)).item())+'\t'+str(np.mean(Sr).item())+'\n')
    fid.close()

# Plot the analysis for different bias intensities
if bias_st == 1:
    with open(dir+'bias_rms=%d.log'%(rms), 'r') as f:
        data_bias = np.zeros([len(f.readlines()),5])
        ib = 0
        f.seek(0,0)
        for line in f:
            data_bias[ib,:] = np.array(str.split(line))
            ib = ib + 1
    plt.figure(1, figsize=(12,4))
    plt.subplot(121)
    plt.scatter(data_bias[:,0],data_bias[:,1],c='g',label='Error of all orders')
    plt.scatter(data_bias[:,0],data_bias[:,2],c='r',label='Error of order %d'%(bias_pos))
    plt.scatter(data_bias[:,0],data_bias[:,3],c='b',label='Error of other order')
    plt.ylim(0,0.5)
    plt.xlabel("Focus Error")
    plt.ylabel("RMS")
    plt.legend()
    plt.title("RMS under different bias conditions for Phase varience=%d"%(rms))

    plt.subplot(122)
    plt.scatter(data_bias[:,0],data_bias[:,4],label='SR')
    plt.ylim(0,1)
    plt.xlabel("Focus Error")
    plt.ylabel("SR")
    plt.legend()
    plt.title("SR under different bias conditions for Phase varience=%d"%(rms))

    plt.savefig(dir+"biasFORrms=%d.png"%(rms),bbox_inches='tight')
    plt.close()
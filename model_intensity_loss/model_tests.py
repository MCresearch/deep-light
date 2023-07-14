#导入包
import numpy as np
import sys
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,MinMaxScaler, StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import Dataset,DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
import sys
from Zernike import *
from fun import *
import time
import json
from propagation import *
import random
from Xception import *
import os
time_start=time.time()
nsnapshot = 1000
model_path ="/home/xianyuer/data/intensityloss_git/model/0711_9y3_128_intloss_intmean_rms4_200000_b16_e50__step_50_lr_0.0001.pt"
dir = "/home/xianyuer/data/intensityloss_git/test_model/0711_9y3_128_intloss_intmean_rms4_200000_b16_e50__step_50_lr_0.0001/"

intensity_dir = "/home/xianyuer/data/intensityloss_git/zer9/5_new_1000_noise_out_0.0001.npy"
zernike_dir = "/home/xianyuer/data/intensityloss_git/zer9/5_1000_zernike_y3.npy"

test_file_name = dir+"zernike_test_real.txt"
test_file_name_p = dir+"zernike_test_predict.txt"
test_file_name_diff = dir+"zernike_test_diff.txt"

x= np.load(intensity_dir)
x =x.astype(np.float32)
y = np.load(zernike_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
if os.path.exists(dir):
    print("文件已经存在")
else:
    os.mkdir(dir)
    
print(np.shape(x))
print(np.shape(y))

print("max of x = ", np.max(x))
print("nsnapshot = %s" % nsnapshot)
print("x shape = ", np.shape(x))
print("y shape = ", np.shape(y))

x = x[:nsnapshot,:,:]
test_y  = y[:nsnapshot]  
test_yp = np.zeros((nsnapshot,7))

print("test_x shape = ", np.shape(x))
print("test_y shape = ", np.shape(test_y))

net = Xception()
net.load_state_dict(torch.load(model_path))
net = net.to(device)

for i in range(nsnapshot):
    x1 = x[i,:,:]
    x1 = torch.tensor(x1).to(device)
    test_x_0  = torch.reshape(x1, [1, 1, 128, 128]).to(device)  
    test_yp_0 = net(test_x_0)
    test_yp[i,:] = test_yp_0.detach().cpu().numpy()

time_end = time.time()
print("time cost: ", time_end-time_start, "s")

np.savetxt(test_file_name, test_y)
np.savetxt(test_file_name_p, test_yp)
np.savetxt(test_file_name_diff, test_y-test_yp)

with open("INPUT_propagation_zer9.json", 'r', encoding='utf-8') as fw:
    injson = json.load(fw)
mm = injson['data']['mm'] 
mgs = injson['data']['mgs']
a0 = injson['data']['a0']
xx0 = injson['data']['xx0']
plm = injson['data']['plm']
zfh = injson['data']['zfh']
xxz = injson['data']['xxz']
maxZnkOrder = injson['data']['maxZnkOrder']
minZnkDim = injson['data']['minZnkDim']
rms = injson['data']['rms']
eeznk = injson['data']['eeznk']

test_ydiff = test_y-test_yp
test_yp = torch.tensor(test_yp).to(device)
test_ydiff = torch.tensor(test_ydiff).to(device)

Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss = parameter(mm,mgs,a0,xx0,plm,zfh,xxz,maxZnkOrder,minZnkDim,rms,eeznk,zernike_dir)

x_pre = sumnor_progagtion(nsnapshot,ngrid,ngrid2,init_intens,test_yp,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz)
x_diff = sumnor_progagtion(nsnapshot,ngrid,ngrid2,init_intens,test_ydiff,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz)

np.save(dir+"pre_sumnor_outintensity.npy",x_pre.detach().cpu().numpy().astype(np.float16))
np.save(dir+"diff_sumnor_outintensity.npy",x_diff.detach().cpu().numpy().astype(np.float16))
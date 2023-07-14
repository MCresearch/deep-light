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
sys.path.append("/home/xianyuer/data/intensityloss_git")
from torchinfo import summary
import matplotlib.pyplot as plt
import sys
from Zernike import *
from fun import *
import time
import json
from propagation import *
import random
# from Xception import *
from Xception_model_256_2 import *
import os
time_start=time.time()
nsnapshot = 2
model_path ="/home/xianyuer/data/intensityloss_git/model/0620_35_128_channel2_intloss_rms4_b16_e500000_lr_0.0001.pt"
dir = "/home/xianyuer/data/intensityloss_git/test_model/0620_35_128_channel2_intloss_rms4_b16_e500000_lr_0.0001/"

intensity_dir = "/home/xianyuer/data/intensityloss_git/5_5_y0__sumnor_outintensity.npy"
zernike_dir = "/home/xianyuer/data/intensityloss_git/5_5_y3_zernike.npy"
intensity_dir2 = "/home/xianyuer/data/intensityloss_git/5_5_y1__sumnor_outintensity.npy"
test_file_name_p = dir+"1_zernike_test_predict.txt"

x1= np.load(intensity_dir)
x2= np.load(intensity_dir2)

x1 =x1.astype(np.float32)
x2 =x2.astype(np.float32)
# y = np.load(zernike_dir)
# y = y[:,2:35]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
if os.path.exists(dir):
    print("文件已经存在")
else:
    os.mkdir(dir)
    
# print(np.shape(x))

# print("max of x = ", np.max(x))
# print("nsnapshot = %s" % nsnapshot)
# print("x shape = ", np.shape(x))

# x = x[:nsnapshot,:,:]
test_yp = np.zeros((nsnapshot,33))

# print("test_x shape = ", np.shape(x))

net = Xception()
net.load_state_dict(torch.load(model_path))
net = net.to(device)


for i in range(nsnapshot):
    x11 = x1[i,:,:]
    x22 = x2[i,:,:]
    x11 = torch.tensor(x11).to(device)
    x22 = torch.tensor(x22).to(device)
    test_x_11  = torch.reshape(x11, [1, 1, 128, 128]).to(device) 
    test_x_22  = torch.reshape(x22, [1, 1, 128, 128]).to(device) 
    test_x_0 = torch.zeros((1,2,128,128)).to(device)
    test_x_0[:,0,:,:] = test_x_11
    test_x_0[:,1,:,:] = test_x_22
    test_x_0.requires_grad_(True)
    print(test_x_0.requires_grad) 
    test_yp_0 = net(test_x_0)
    set = torch.zeros_like(test_yp_0)
    set[0,3] = 1
    test_yp_0.backward(set)
    aa = test_x_0.grad
    # test_yp_0.backward()
    # aa = test_x_0.grad
    #     # test_yp[i,:] = test_yp_0.detach().cpu().numpy()
    print(np.shape(aa))
    plt.figure(1,dpi=600)
    plt.contourf(aa.detach().cpu().numpy()[0,0,:,:],levels=100, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.savefig((dir+str(i)+"_simple2_zernike6.png"),bbox_inches='tight')
    plt.close()
time_end = time.time()
print("time cost: ", time_end-time_start, "s")
# for i in range(4):
#     print("rms",i+1,":",np.sqrt(np.mean(np.power(test_yp[i*1000:(i+1)*1000,:]-test_y[i*1000:(i+1)*1000,:], 2))))

    
# np.savetxt(test_file_name, test_y)
np.savetxt(test_file_name_p, test_yp)
# np.savetxt(test_file_name_diff, test_y-test_yp)
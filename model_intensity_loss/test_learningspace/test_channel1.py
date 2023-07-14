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
from Xception import *
# from Xception_model_256_2 import *
import os
time_start=time.time()
nsnapshot = 2
nzernike = 9
m_grid = 128
rms = 4
aaz = 0.2196/2
model_name = "0711_9y3_128_intloss_intmean_rms4_200000_b16_e50__step_50_lr_0.0001"
model_path ="/home/xianyuer/data/intensityloss_git/model/"+model_name+".pt"
dir = "/home/xianyuer/data/intensityloss_git/test_model/"+model_name+"/"

intensity_dir = "/home/xianyuer/data/intensityloss_git/zer9/5_1000_sumnor_outintensity_y3.npy"
zernike_dir = "/home/xianyuer/data/intensityloss_git/zer9/5_1000_zernike_y3.npy"
test_file_name_p = dir+"1_zernike_test_predict.txt"

x= np.load(intensity_dir)
x1 =x.astype(np.float32)
# y = np.load(zernike_dir)
# y = y[:,2:35]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
if os.path.exists(dir):
    print("文件已经存在")
else:
    os.mkdir(dir)
    
print(np.shape(x))

print("max of x = ", np.max(x))
print("nsnapshot = %s" % nsnapshot)
print("x shape = ", np.shape(x))

x = x[:nsnapshot,:,:]
test_yp = np.zeros((nsnapshot,nzernike-2))

# print("test_x shape = ", np.shape(x))

net = Xception()
net.load_state_dict(torch.load(model_path))
net = net.to(device)


for i in range(nsnapshot):
    x1 = x[i,:,:]
    x1 = torch.tensor(x1).to(device)
    test_x_0  = torch.reshape(x1, [1, 1, 128, 128]).to(device) 
    test_x_0 =test_x_0.float()
    test_x_0.requires_grad_(True)
    print(test_x_0.requires_grad) 
    test_yp_0 = net(test_x_0)
    set = torch.zeros_like(test_yp_0)
    set[0,0] = 1
    test_yp_0.backward(set)
    aa = test_x_0.grad
    # test_yp_0.backward()
    # aa = test_x_0.grad
    #     # test_yp[i,:] = test_yp_0.detach().cpu().numpy()
    print(np.shape(aa))
    plt.figure(1,dpi=300)
    plt.contourf(aa.detach().cpu().numpy()[0,0,:,:],levels=100, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.yticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig((dir+str(i)+"_simple2_zernike3.png"),bbox_inches='tight')
    plt.close()
time_end = time.time()
print("time cost: ", time_end-time_start, "s")
# for i in range(4):
#     print("rms",i+1,":",np.sqrt(np.mean(np.power(test_yp[i*1000:(i+1)*1000,:]-test_y[i*1000:(i+1)*1000,:], 2))))

    
# np.savetxt(test_file_name, test_y)
np.savetxt(test_file_name_p, test_yp)
# np.savetxt(test_file_name_diff, test_y-test_yp)
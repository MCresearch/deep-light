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
sys.path.append("..")
import matplotlib.pyplot as plt
import sys
from Zernike import *
from fun import *
import time
import json
from propagation_speed9 import *
import random
from Xception_model import *
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

model_name = "test"
if os.path.exists(model_name):
    print("文件已经存在")
else:
    os.mkdir(model_name)
    
dir = "./"+model_name+"/"
print(dir)

model_path = "/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/model/0228_35_3456789_64_100_b2_gauss1loss_rms4_changeloss_e200000_lr0.0002.pt"
batch_size = 1
#batch_size=8
seed = 12333345
data_time = "220223"
input_model = False
mgs_guass = 1
snapshoot = 1000
nzer = 7
with open("INPUT_64.json", 'r', encoding='utf-8') as fw:
        injson = json.load(fw)
mm = injson['data']['mm'] 
mgs = injson['data']['mgs']
a0 = injson['data']['a0']
xx0 = injson['data']['xx0']
plm = injson['data']['plm']
zfh = injson['data']['zfh']
xxz = injson['data']['xxz']
minZnkDim = injson['data']['minZnkDim']
maxZnkOrder = injson['data']['maxZnkOrder']
rms = injson['data']['rms']
eeznk = injson['data']['eeznk']
zernike_dir = injson['data']['dir']
maxZnkOrder = 3
xxz = 25

Zernike_alias = np.array([-1] * 3 + [1] * 4, dtype=np.float32)
Zernike_alias  = torch.tensor(Zernike_alias).to(device)
Zer = Zer1(maxZnkOrder,mm,a0,xx0)
Zer = torch.tensor(Zer).to(device)
init_intens = init_intensity(mm,a0,xx0,mgs)
init_intens = torch.tensor(init_intens).to(device)

nxzz = a0*xx0
ngrid = pow(2,mm)
n1 = ngrid/2 + 1
aa0 = xx0*a0
dxy0 = aa0/ngrid
airy = 1.22*plm*zfh/(2*a0)
aaz = airy*xxz
dxyz = aaz/ngrid
ngrid2 = ngrid//2
a02 = a0*a0
gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1,ngrid-n1,ngrid),dxyz*np.linspace(1-n1,ngrid-n1,ngrid))

dlta = (1-aaz/aa0)/zfh
ddxz = 1-dlta*zfh
dk0 = 1/aa0
zzzz = zfh/(1-dlta*zfh)
wave_number = 2*torch.pi/plm
ei = -wave_number*(gx*gx+gy*gy)/2*(1/zfh)
ei = torch.tensor(ei).to(device)
ec = wave_number*gx*gx*dlta/2 + wave_number*gy*gy*dlta/2
ec = torch.tensor(ec).to(device)
h = torch.zeros(ngrid).to(device)
h_sum = torch.zeros((ngrid,ngrid)).to(device)
prop1(ngrid,n1,zzzz,wave_number,aa0,h)
for i in range(ngrid):
    for j in range(ngrid):
        h_sum[i,j] = h[i]+h[j]
        
ez = -1*wave_number*gx2*gx2*dlta/(2*ddxz) - wave_number*gy2*gy2*dlta/(2*ddxz)
ez = torch.tensor(ez).to(device)
mask0 = ((gx**2 + gy**2)/a02 <= 1)
mask0 = torch.tensor(mask0).to(device)
f_m = torch.exp(1j*ei)*torch.exp(1j*ec)

# gauss矩阵
r02 = a0*a0*xx0*xx0/9
gauss = np.zeros((ngrid,ngrid))
mask00 = ((gx*gx + gy*gy)<=r02)
a = np.exp(-1*pow(((gx*gx+gy*gy)/a02),mgs_guass)) 
gauss  = a * mask00
gauss  = torch.tensor(gauss).to(device)

net = Xception()
net.load_state_dict(torch.load(model_path))
net = net.to(device)


loss_int= torch.zeros([snapshoot,1],dtype=torch.float).to(device)  
loss_zer= torch.zeros([snapshoot,nzer],dtype=torch.float).to(device)
y_predict_choose = torch.zeros([snapshoot,nzer],dtype=torch.float).to(device)
rms_save = torch.zeros([snapshoot,2],dtype=torch.float).to(device)
x_diff = torch.zeros([snapshoot, 64, 64],dtype=torch.float).to(device)
y_test = cc(snapshoot,maxZnkOrder,"random",eeznk,rms,zernike_dir)
y_test = y_test[:,2:]
y_test = torch.tensor(y_test).to(device)
        
for i in range(snapshoot):

    c_test = torch.reshape(y_test[i,:], [1, nzer]).to(device)  
    x_test = nor_progagtion(batch_size,ngrid,ngrid2,init_intens,c_test,Zer,mask0,f_m,h_sum,ez,ddxz)
    x_test = torch.reshape(x_test, [batch_size, 1, 64, 64]).to(device)  

    y_predict = net(x_test.reshape([batch_size,1,64,64]))
        
    x_predict =  nor_progagtion(batch_size,ngrid,ngrid2,init_intens,y_predict,Zer,mask0,f_m,h_sum,ez,ddxz)
    
    loss_z1 = torch.mean(pow(y_predict[0,:] - c_test[0,:],2))
    loss_z2 = torch.mean(pow(y_predict[0,:]*Zernike_alias - c_test[0,:],2))
    if loss_z1>loss_z2:
        y_predict_choose[i,:] = y_predict[0,:] * Zernike_alias
    else:
        y_predict_choose[i,:] = y_predict[0,:]
          
    loss_int[i]= torch.mean(pow((x_predict - x_test[:,0,:,:]),2))  
    rms_save[i,0] = torch.sum(pow(c_test-y_predict[0,:],2))
    rms_save[i,1] = torch.sum(pow(c_test-y_predict_choose[i,:],2))
    x_diff =  nor_progagtion(batch_size,ngrid,ngrid2,init_intens,c_test-y_predict_choose[i,:],Zer,mask0,f_m,h_sum,ez,ddxz)
    if rms_save[i,1] > 2:
        print("bad predict:rms=",rms_save[i,1])
        plt.figure(1, figsize=(12,4))
        plt.subplot(121)
        plt.bar(np.array(range(7)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
        plt.bar(np.array(range(7)),y_predict.detach().cpu().numpy()[0,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
        plt.xlabel("Zernike order",fontsize=15)
        plt.ylabel("Zernike coefficient values",fontsize=15)
        plt.xticks(size = 10)
        plt.yticks(size=10)
        # plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
        plt.legend()

        plt.subplot(122)
        plt.bar(np.array(range(7)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
        plt.bar(np.array(range(7)),y_predict_choose.detach().cpu().numpy()[i,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
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
plt.bar(np.array(range(7)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
plt.bar(np.array(range(7)),y_predict.detach().cpu().numpy()[0,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
plt.xlabel("Zernike order",fontsize=15)
plt.ylabel("Zernike coefficient values",fontsize=15)
plt.xticks(size = 10)
plt.yticks(size=10)
# plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
plt.legend()

plt.subplot(122)
plt.bar(np.array(range(7)),c_test.detach().cpu().numpy()[0,:], color="red",alpha=1,label = "Initial values")
plt.bar(np.array(range(7)),y_predict_choose.detach().cpu().numpy()[snapshoot-1,:], color="blue",alpha=0.5,label = "Predict(by Xception)")
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
#导入包
import numpy as np
import sys
# import sklearn
from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from propagation import *
import random
from Xception import *
from model_fit import *
#https://blog.csdn.net/BernardDong/article/details/125495796
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# Parameters setting 
with open("INPUT_model_test.json", 'r', encoding='utf-8') as fw:
    injson_model = json.load(fw)

model_name = injson_model['model']['name']
model_path = injson_model['model']['model_path']
epoch = injson_model['model']['epoch']
batch_size = injson_model['model']['batch_size']
lr = injson_model['model']['lr']
seed = injson_model['model']['seed']
dir = injson_model['model']['dir']
loss_type = injson_model['model']['loss_type']
save = injson_model['model']['save']
data_train = injson_model['model']['data_train']
nsnapshot = injson_model['model']['nsnapshot']
print_step = injson_model['model']['print_step']
save_step = injson_model['model']['save_step']

with open("INPUT_propagation_test.json", 'r', encoding='utf-8') as fw:
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
zernike_dir = injson['data']['zernike_dir']

# Load model
net = Xception()
if model_path == "False":
    net = net.to(device)
else:
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    
# Transmission parameter calculation
Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss = parameter(mm,mgs,a0,xx0,plm,zfh,xxz,maxZnkOrder,minZnkDim,rms,eeznk,zernike_dir)
# Set the random number seed
torch.manual_seed(51)

if data_train == "random": 
    # Model fit
    fit(model_name,net,loss_type,save,batch_size,epoch,lr,print_step,save_step,zernike_dir,Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss)

if data_train == "confirm":
    dir1 = "/home/xianyuer/data/intensityloss_git/"
    intensity_dir1 = dir1+"1_50000y3_sumnor_outintensity.npy"
    zernike_dir1 = dir1+"1_50000y3_zernike.npy"

    # intensity_dir2 =dir1+"2_50000y3_sumnor_outintensity.npyy"
    # zernike_dir2 =dir1+"2_50000y3_zernike.npy"

    # intensity_dir3 =dir1+"3_50000y3_sumnor_outintensity.npy"
    # zernike_dir3 =dir1+"3_50000y3_zernike.npy"

    # intensity_dir4 =dir1+"4_50000y3_sumnor_outintensity.npy"
    # zernike_dir4 =dir1+"4_50000y3_zernike.npy"

    # intensity_dir5 =dir1+"5_2000nor_outIntensity_65_0-1_4_2000.npy"
    # zernike_dir5 =dir1+"5_50000y3_zernike.npy"
    x1 = np.load(intensity_dir1)
    y1 = np.load(zernike_dir1)
    # x2 = np.load(intensity_dir2)
    # y2 = np.load(zernike_dir2)
    # x3 = np.load(intensity_dir3)
    # y3 = np.load(zernike_dir3)
    # x4 = np.load(intensity_dir4)
    # y4 = np.load(zernike_dir4)
    # x5 = np.load(intensity_dir5)
    # y5 = np.load(zernike_dir5)
    # y = np.concatenate((y1,y2))
    # y = np.concatenate((y,y3))
    # y = np.concatenate((y,y4))

    # x = np.concatenate((x1,x2))
    # x = np.concatenate((x,x3))
    # x = np.concatenate((x,x4))
    x = x1
    y = y1
    print("x shape = ", np.shape(x))
    print("y shape = ", np.shape(y))
    print("nsnapshot = %s" % nsnapshot)
    train_x = x[:nsnapshot].reshape([nsnapshot, 1, ngrid, ngrid])
    train_y = y[:nsnapshot]
    x_train = train_x 
    y_train = train_y 
    # x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.2)
    print("x_train shape = ", np.shape(x_train))
    print("y_train shape = ", np.shape(y_train))
    # print("x_test shape = ", np.shape(x_test))
    # print("y_test shape = ", np.shape(y_test))
    x_train = torch.tensor(x_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    # x_test = torch.tensor(x_test).to(device)
    # y_test = torch.tensor(y_test).to(device)
    
    # 构建Dataset数据集
    class MyDataset(Dataset):#需要继承torch.utils.data.Dataset
        def __init__(self,feature,target):
            super(MyDataset, self).__init__()
            self.feature =feature
            self.target = target
        def __getitem__(self,index):
            item=self.feature[index]
            label=self.target[index]
            return item,label
        def __len__(self):
            return len(self.feature)
    # 封装成DataLoader对象
    bs = batch_size 
    train_data=MyDataset(x_train,y_train)
    train_data=DataLoader(train_data, batch_size=bs, shuffle=True)
    
    fit2(train_data,model_name,net,loss_type,save,batch_size,epoch,lr,print_step,save_step,zernike_dir,Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss)
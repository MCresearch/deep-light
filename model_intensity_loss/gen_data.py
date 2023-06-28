#导入包
import numpy as np
import sys
# import sklearn
# from sklearn.model_selection import train_test_split
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
nsnapshot = 50000
dir = "4_50000"

with open("INPUT_propagation.json", 'r', encoding='utf-8') as fw:
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
zernike_dir = "/home/xianyuer/data/intensityloss_git/4_50000zernike.npy"

# Transmission parameter calculation
Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss = parameter(mm,mgs,a0,xx0,plm,zfh,xxz,maxZnkOrder,minZnkDim,rms,eeznk,zernike_dir)
# Set the random number seed
if zernike_dir =="random": 
    y = cc(nsnapshot,maxZnkOrder,"random",eeznk,rms,zernike_dir)
    y = y[:,2:]
else:
    y = np.load(zernike_dir)
y[:,0] = y[:,0]+3  
y = torch.tensor(y).to(device)
x = sumnor_progagtion(nsnapshot,ngrid,ngrid2,init_intens,y,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz)
np.save(dir+"y4_sumnor_outintensity.npy",x.detach().cpu().numpy().astype(np.float16))
np.save(dir+"y4_zernike.npy",y.detach().cpu().numpy().astype(np.float16))
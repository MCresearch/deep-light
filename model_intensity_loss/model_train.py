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
with open("INPUT_model.json", 'r', encoding='utf-8') as fw:
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
zernike_dir = injson['data']['zernike_dir']

# Transmission parameter calculation
Zer,maxZnkDim = Zer1(maxZnkOrder,mm,a0,xx0)
print(maxZnkDim)

Zernike_alias_all = np.array([1] * 2 + [-1] * 3 + [1] * 4 + [-1] * 5 + [1] * 6 + [-1] * 7 + [1] * 8+ [-1] * 9+ [1] * 10 + [-1] * 11+ [1] * 12+ [-1] * 13 + [1] * 14, dtype=np.float32)
Zernike_alias =  Zernike_alias_all[2:maxZnkDim]
Zernike_alias  = torch.tensor(Zernike_alias).to(device)
Zer,maxZnkDim = Zer1(maxZnkOrder,mm,a0,xx0)
print("shape zer",np.shape(Zer))
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

# gauss
r02 = a0*a0*xx0*xx0/36
gauss = np.ones((ngrid,ngrid))
mask00 = ((gx*gx + gy*gy)<=r02)
gauss  = gauss * mask00
gauss  = torch.tensor(gauss).to(device)
gauss = torch.reshape(gauss, [1, ngrid, ngrid]).to(device)  
print(gauss)

# Set the random number seed
torch.manual_seed(51)

# Load model
net = Xception()
if model_path == "False":
    net = net.to(device)
else:
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
# Model fit
fit(model_name,net,loss_type,save,batch_size,epoch,lr,zernike_dir,Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss)

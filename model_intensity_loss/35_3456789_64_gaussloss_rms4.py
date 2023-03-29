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
#https://blog.csdn.net/BernardDong/article/details/125495796
######### parameters setting ########
model_name = "0322_35_128_200_64_4_8" # date_ZernikeOrder_ngrid(d2for downsample)_epochs_batch_rms_learningrate(3to8:1e-3~1e-8)
model_path = "./model/0322_35_128_400_32_4_7.pt"
epoch = 200
batch_size = 64
#batch_size=8
seed = 12333345
data_time = "220223"
input_model = False
dir = "./"
mgs_guass = 0
#####################################
fid = open(model_name+'.log', 'w')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# 加载数据
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
maxZnkOrder = 7
xxz = 36

Zernike_alias = np.array([-1] * 3 + [1] * 4 + [-1] * 5 + [1] * 6 + [-1] * 7 + [1] * 8, dtype=np.float32)
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

def fit(net,batch_size,epochs,learning_rate,Zernike_alias,ngrid,ngrid2,init_intens,Zer,mask0,f_m,h_sum,ez,ddxz,gauss):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate) 
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = epochs/5, gamma=0.1)
    Zernike_alias_0 = torch.zeros((batch_size,33)).to(device)  
    for i in range(batch_size):
        Zernike_alias_0[i,:] = Zernike_alias
    #损失函数
    criterion=torch.nn.MSELoss(reduction='mean')  
    # 监视进度
    samples = 0
    # 监视准确度
    corrects = 0
    # 全数据训练几次
    for epoch in range(epochs):
        # 对每个batch进行训练
        # 生成数据
        y = cc(batch_size,maxZnkOrder,"random",eeznk,rms,zernike_dir)
        y = y[:,2:]
        y = torch.tensor(y).to(device)
        x = nor_progagtion(batch_size,ngrid,ngrid2,init_intens,y,Zer,mask0,f_m,h_sum,ez,ddxz)
        x = torch.reshape(x, [batch_size, 1,128, 128]).to(device)  
        # 正向传播
        cz_pred = net(x)
        # change loss
        # loss_z1 = torch.sum(pow(cz_pred - y,2))
        # loss_z2 = torch.sum(pow(cz_pred*Zernike_alias - y,2))
        # if loss_z1>loss_z2:
        #     cz_pred = cz_pred * Zernike_alias
        # 计算损失
        # far_field_intens_pred =  nor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,mask0,f_m,h_sum,ez,ddxz)
        # far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, 128, 128]).to(device)  
        # loss = torch.mean(pow((far_field_intens_pred - x)*gauss,2))
        loss = criterion(cz_pred.float(),y.float())
        # 梯度清零
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 更新梯度
        opt.step()
        
        # 监视进度：每训练一个batch，模型见过的数据就会增加x.shape[0]
        samples += x.shape[0]
        #每100个epoch和最后结束时，打印模型的进度
        if (epoch + 1) % (100) == 0 or epoch  == (epochs - 1):
            loss_z1 = torch.mean(pow(cz_pred - y,2))
            loss_z2 = torch.mean(pow(cz_pred*Zernike_alias_0 - y,2))
            loss_zernike = np.minimum(loss_z1.detach().cpu().numpy(), loss_z2.detach().cpu().numpy()) 
            # loss_zernike = np.mean(pow(cz_pred.detach().cpu().numpy() - y.detach().cpu().numpy(),2))
            # 监督模型进度
            print("Epoch{}:[{}/{} {: .0f}%], Loss:{:.6f} ".format(
                epoch + 1
                , samples
                , epochs*batch_size
                , 100*samples/(epochs*batch_size)
                , loss.data.item()),"Zer loss: ",loss_zernike)
            fid.write(str(loss.cpu().item())+'\t'+str(loss_zernike))
            fid.write('\n')

    torch.save(net.state_dict(), './model/'+model_name+'.pt')

# 设置随机种子
torch.manual_seed(51)

# 实例化模型
net = Xception()
net.load_state_dict(torch.load(model_path))
net = net.to(device)
# 学习率
lr = 0.00000001

fit(net, batch_size,epoch,lr,Zernike_alias,ngrid,ngrid2,init_intens,Zer,mask0,f_m,h_sum,ez,ddxz,gauss)


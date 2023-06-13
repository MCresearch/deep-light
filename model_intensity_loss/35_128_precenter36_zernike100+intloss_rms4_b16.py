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
from propagation_speed9_sumnor import *
import random
from Xception_model_256 import *
#https://blog.csdn.net/BernardDong/article/details/125495796
######### parameters setting ########
model_name = "0529_35_128_precenter36_zernike100+intloss_rms4_b16_e500000_lr_"
# model_path = "/data/home/scv9267/run/1_zhangxianyue/deeplight/model_intloss/0323_35_256_128_precenter36+zernikeloss_x_rms4_b16_e1000000_lr_0.0001.pt"
epoch = 500000
batch_size = 16
#batch_size=8
seed = 12333345
data_time = "220529"
input_model = False
dir = "./"
mgs_guass = 1
lr = 0.0001
#####################################
fid = open(model_name+str(lr)+'.log', 'w')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# 加载数据
with open("INPUT_128.json", 'r', encoding='utf-8') as fw:
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

Zernike_alias = np.array([-1] * 3 + [1] * 4 + [-1] * 5 + [1] * 6 + [-1] * 7 + [1] * 8, dtype=np.float32)
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

# pool = torch.nn.MaxPool2d(
#         2, 
#         stride=None, 
#         padding=0, 
#         dilation=1, 
#         return_indices=False, 
#         ceil_mode=False)

# gauss矩阵
r02 = a0*a0*xx0*xx0/36
gauss = np.ones((ngrid,ngrid))
mask00 = ((gx*gx + gy*gy)<=r02)
# a = np.exp(-1*pow(((gx*gx+gy*gy)/a02),mgs_guass)) 
gauss  = gauss * mask00
gauss  = torch.tensor(gauss).to(device)
gauss = torch.reshape(gauss, [1, ngrid, ngrid]).to(device)  
print(gauss)
# gauss_down = pool(gauss)

plt.figure(1, dpi=300)
plt.contourf(gauss .detach().cpu().numpy()[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
plt.colorbar()
# plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
# plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)
plt.savefig(dir+"_gauss.png",bbox_inches='tight')
plt.close()

def fit(net,batch_size,epochs,learning_rate,Zernike_alias,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss_down):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate) 
    Zernike_alias_0 = torch.zeros((batch_size,maxZnkDim-2)).to(device)  
    gauss_0 = torch.zeros((batch_size,1,ngrid,ngrid)).to(device)  
    loss_z1 =  torch.zeros((batch_size)).to(device) 
    loss_z2 =  torch.zeros((batch_size)).to(device) 
    loss_zer =  torch.zeros((batch_size)).to(device)
    for i in range(batch_size):
        Zernike_alias_0[i,:] = Zernike_alias
        gauss_0[i,0,:,:] = gauss_down
    #降采样函数
    # pool = torch.nn.MaxPool2d(
    #     2, 
    #     stride=None, 
    #     padding=0, 
    #     dilation=1, 
    #     return_indices=False, 
    #     ceil_mode=False)
    # scheduler = optim.lr_scheduler.LinearLR(opt,start_factor=0.00001, total_iters=epochs)
    #损失函数
    criterion = torch.nn.MSELoss(reduction='mean')  
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
        x = sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,y,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz)
        # x_down = pool(x)
        x = torch.reshape(x, [batch_size, 1, ngrid, ngrid]).to(device)  
        x = x * gauss_0
        # 正向传播
        cz_pred = net(x)

        # # 计算损失
        far_field_intens_pred =  sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz)
        # far_field_intens_pred_down = pool(x)
        far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, ngrid, ngrid]).to(device)  
        far_field_intens_pred = far_field_intens_pred * gauss_0
        loss_int = torch.mean(pow(far_field_intens_pred - x,2))
        
        cz_pred = cz_pred.to(torch.float32)
        y = y.to(torch.float32)
        loss_zer = criterion(cz_pred,y)
        loss = loss_int + loss_zer/100
        # 梯度清零
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 更新梯度
        opt.step()
        # print(f"当前学习率：{opt.param_groups[0]['lr']}")
        # scheduler.step()
        
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

    torch.save(net.state_dict(), './model/'+model_name+str(learning_rate)+'.pt')

# 设置随机种子
torch.manual_seed(51)

# 实例化模型
net = Xception()
# net.load_state_dict(torch.load(model_path))
net = net.to(device)
# 学习率
fit(net,batch_size,epoch,lr,Zernike_alias,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss)


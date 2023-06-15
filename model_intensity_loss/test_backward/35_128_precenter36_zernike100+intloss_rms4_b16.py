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
#https://blog.csdn.net/BernardDong/article/details/125495796
######### parameters setting ########
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
#####################################
fid = open(model_name+str(lr)+'.log', 'w')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# 加载数据
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

# gauss矩阵
r02 = a0*a0*xx0*xx0/36
gauss = np.ones((ngrid,ngrid))
mask00 = ((gx*gx + gy*gy)<=r02)
gauss  = gauss * mask00
gauss  = torch.tensor(gauss).to(device)
gauss = torch.reshape(gauss, [1, ngrid, ngrid]).to(device)  
print(gauss)

plt.figure(1, dpi=300)
plt.contourf(gauss .detach().cpu().numpy()[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)
plt.savefig(dir+"_gauss.png",bbox_inches='tight')
plt.close()

def fit(net,loss_type,save,batch_size,epochs,learning_rate,zernike_dir,Zernike_alias,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss_down):
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

    #损失函数
    criterion = torch.nn.MSELoss(reduction='mean')  
    # 监视进度
    samples = 0
    # 监视准确度
    corrects = 0
    
    cz_pred_all = torch.zeros((epochs,maxZnkDim-2))
    back = torch.zeros((epochs,ngrid,ngrid))
    back_00 = torch.zeros((epochs))
    # 全数据训练几次
    if zernike_dir =="random": 
        y = cc(batch_size,maxZnkOrder,"random",eeznk,rms,zernike_dir)
        y = y[:,2:]
    else:
        y = np.loadtxt(zernike_dir)
    y = torch.tensor(y).to(device)
    x = sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,y,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz)
    x_c = x.clone()
    if save == "Ture":
        np.save(dir+"sumnor_outintensity_5050_25.npy",x_c.detach().cpu().numpy())
    for epoch in range(epochs):
        x1 = x.clone()
        x1[0,50,50] = x[0,50,50]+0.00001*epoch
        x1 = torch.reshape(x1, [batch_size, 1, ngrid, ngrid]).to(device)  
        x1.requires_grad_(True)
        # 正向传播
        cz_pred = net(x1)
        cz_pred_all[epoch,:] = cz_pred.clone()
        
        #  计算损失
        if loss_type == "Zernikeloss":
            cz_pred = cz_pred.to(torch.float32)
            y = y.to(torch.float32)
            loss_zer = criterion(cz_pred,y)
            loss = loss_zer
        elif loss_type == "intensityloss":
            far_field_intens_pred =  sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz)
            far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, ngrid, ngrid]).to(device)  
            far_field_intens_pred = far_field_intens_pred * gauss_0
            loss_int = torch.mean(pow(far_field_intens_pred - x,2))
            loss = loss_int
        elif loss_type == "Zernike+intensityloss":  
            cz_pred = cz_pred.to(torch.float32)
            y = y.to(torch.float32)
            loss_zer = criterion(cz_pred,y)
            far_field_intens_pred =  sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz)
            far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, ngrid, ngrid]).to(device)  
            far_field_intens_pred = far_field_intens_pred * gauss_0
            loss_int = torch.mean(pow(far_field_intens_pred - x,2))
            loss = loss_int + loss_zer
            
        else:
            print("error loss_type")
       
        set = torch.zeros_like(cz_pred)
        set[0,25] = 1
        cz_pred.backward(set)
        back[epoch,:,:] = x1.grad
        print(x1.grad)
        # loss.backward()
        # 更新梯度
        # opt.step()
        
        # # 监视进度：每训练一个batch，模型见过的数据就会增加x.shape[0]
        # samples += x.shape[0]
        # #每100个epoch和最后结束时，打印模型的进度
        # if (epoch + 1) % (100) == 0 or epoch  == (epochs - 1):
        #     loss_z1 = torch.mean(pow(cz_pred - y,2))
        #     loss_z2 = torch.mean(pow(cz_pred*Zernike_alias_0 - y,2))
        #     loss_z1_c = loss_z1.clone()
        #     loss_z2_c = loss_z2.clone()
        #     loss_zernike = np.minimum(loss_z1_c.detach().cpu().numpy(), loss_z2_c.detach().cpu().numpy()) 
        #     # loss_zernike = np.mean(pow(cz_pred.detach().cpu().numpy() - y.detach().cpu().numpy(),2))
        #     # 监督模型进度
        #     print("Epoch{}:[{}/{} {: .0f}%], Loss:{:.6f} ".format(
        #         epoch + 1
        #         , samples
        #         , epochs*batch_size
        #         , 100*samples/(epochs*batch_size)
        #         , loss.data.item()),"Zer loss: ",loss_zernike)
        #     fid.write(str(loss.cpu().item())+'\t'+str(loss_zernike))
        #     fid.write('\n')
    for i in range(epochs):
        back_00[i] = back[i,50,50]
    np.savetxt("dzernike_5050_25.txt",cz_pred_all.detach().cpu().numpy())
    np.savetxt("dzernike_back_5050_25.txt",back_00.detach().cpu().numpy())
    torch.save(net.state_dict(), model_name+str(learning_rate)+'.pt')

# 设置随机种子
torch.manual_seed(51)

# 实例化模型
net = Xception()
if model_path == "False":
    net = net.to(device)
else:
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
# 学习率
fit(net,loss_type,save,batch_size,epoch,lr,zernike_dir,Zernike_alias,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss)

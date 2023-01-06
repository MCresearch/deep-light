#导入包
import numpy as np
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
sys.path.append("../")
from Zernike import *
from fun import *
import time
import json
from propagation import *
#https://blog.csdn.net/BernardDong/article/details/125495796
######### parameters setting ########
data_size = [2000, 10000, 10000, 17000, 43000, 100000]
model_name = "2000_35_64_10_intloss"
epoch = [50, 100,600, 500, 400, 300]
batch_size = 16
#batch_size=8
seed = 12333345
data_time = "221223"
input_model = False
dir = "./data/"
# input_model = "./35_128_1-50_midloop4_loss_noise_0.05_220823_200000_50.h5"
#model_path = "0921_2_20210322_10_0.1_17000_300.h5"
#####################################
fid = open('2000_35_64_10_intloss.log', 'w')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# 加载数据
# intensity_dir = dir+"nor_outintensity.npy"
# zernike_dir = dir+"zernike_coeff.npy"

intensity_dir1 = dir+"2000_nor_outintensity.npy"
zernike_dir1 = dir+"2000_zernike_coeff.npy"

# x1 = np.load(intensity_dir)
# y1 = np.load(zernike_dir)
x = np.load(intensity_dir1)
y = np.load(zernike_dir1)
y = y[:,2:]

print("x shape = ", np.shape(x))
print("y shape = ", np.shape(y))

for i in range(0, 1):
    nsnapshot = data_size[i]
    nepoch = epoch[i]
    print("nsnapshot = %s" % nsnapshot)
    print("x shape = ", np.shape(x))
    print("y shape = ", np.shape(y))
    train_x = x[:nsnapshot].reshape([nsnapshot,1,64,64])
    train_y = y[:nsnapshot]
    print("train_x shape = ", np.shape(train_x))
    print("train_y shape = ", np.shape(train_y))
    test_x = x[-1000:].reshape([1000, 1, 64, 64])
    test_y = y[-1000:]
    print("test_x shape = ", np.shape(test_x))
    print("test_y shape = ", np.shape(test_y))

# 转换为tensor张量
train_x=torch.tensor(train_x,dtype=torch.float).to(device)
test_x=torch.tensor(test_x,dtype=torch.float).to(device)
train_y=torch.tensor(train_y,dtype=torch.float).to(device)
test_y=torch.tensor(test_y,dtype=torch.float).to(device)

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
train_data=MyDataset(train_x,train_y)
train_data=DataLoader(train_data, batch_size=batch_size, shuffle=True)

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(SeparableConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                               dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, strides=1, start_with_relu=True, channel_change=True):
        super(Block, self).__init__()

        if out_channels != in_channels or strides != 1:
            self.shutcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shutcut = nn.Sequential()

        layers = []
        channels = in_channels

        if channel_change:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(in_channels, out_channels, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            channels = out_channels

        for i in range(num_conv - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(channels, channels, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(channels))

        if not channel_change:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(in_channels, out_channels, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))

        if start_with_relu:
            layers = layers
        else:
            layers = layers[1:]

        if strides != 1:
            layers.append(nn.MaxPool2d(3, strides, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        layer = self.layers(x)
        shutcut = self.shutcut(x)
        out = layer + shutcut
        return out


class Xception(nn.Module):
    def __init__(self, num_classes=33):

        super(Xception, self).__init__()

        self.num_classes = num_classes

        # Entry flow
        self.entry_flow = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            Block(64, 128, 2, 2, start_with_relu=False, channel_change=True),
            Block(128, 256, 2, 2, start_with_relu=True, channel_change=True),
            Block(256, 728, 2, 2, start_with_relu=True, channel_change=True),
        )

        # Middle flow
        self.middle_flow = nn.Sequential(
            Block(728, 728, 3, 1, start_with_relu=True, channel_change=True),
            Block(728, 728, 3, 1, start_with_relu=True, channel_change=True),
            Block(728, 728, 3, 1, start_with_relu=True, channel_change=True),
            Block(728, 728, 3, 1, start_with_relu=True, channel_change=True),
            
        )

        # Exit flow
        self.exit_flow = nn.Sequential(
            Block(728, 1024, 2, 2, start_with_relu=True, channel_change=False),

            SeparableConv(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),

            SeparableConv(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
def fit(net, batchdata, lr,epochs):
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
    dir = injson['data']['dir']
    Phase_option = injson['data']['Phase_option']  ##"random" "confirm" 
    nsnapshot = injson['data']['nsnapshot'] 
    zernike_dir = injson['data']['zernike_dir'] 
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    Zernike_alias = np.array([-1] 
                            + [1] * 2
                            + [-1] * 3
                            + [1] * 4
                            + [-1] * 5
                            + [1] * 6
                            + [-1] * 7
                            + [1] * 8, dtype=np.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)  
    #损失函数
    criterion=torch.nn.MSELoss(reduction='mean')  
    # 监视进度
    samples = 0
    # 监视准确度
    corrects = 0
    # 全数据训练几次
    Zer = Zer1(maxZnkOrder,mm,a0,xx0)
    Zer = torch.tensor(Zer).to(device)
    init_intens = init_intensity(mm,a0,xx0,mgs)
    init_intens = torch.tensor(init_intens).to(device)
    for epoch in range(epochs):
        # 对每个batch进行训练
        for batch_idx, (x,y) in enumerate(batchdata):
            # 正向传播
            cz_pred = net(x)
            # 计算损失
            far_field_intens_pred =  nor_progagtion(batch_size,mm,a0,xx0,plm,zfh,xxz,init_intens,cz_pred,Zer)
            # img1_img = torch.tensor(np.float32(np.expand_dims(far_field_intens_pred, axis=1))).to(device)
            # img1_img.requires_grad_(True)
            # img1_img.retain_grad()
            # print("0",np.shape(x)) 
            far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, 64, 64]).to(device)  
            loss = torch.sum(torch.abs(far_field_intens_pred - x))
        
            # loss.requires_grad_(True)
            # loss = criterion(cz_pred, y)
            # 反向传播
            loss.backward()
            # print(cz_pred.grad)
            # 更新梯度
            opt.step()
            # 梯度清零
            opt.zero_grad()
            
            # 监视进度：每训练一个batch，模型见过的数据就会增加x.shape[0]
            samples += x.shape[0]
            #每200个batch和最后结束时，打印模型的进度
            
            if (batch_idx + 1) % nsnapshot == 0 or batch_idx == (len(batchdata) - 1):
                loss_z1 = np.sum(np.abs(cz_pred.detach().cpu().numpy()[batch_size-1,:] - y.detach().cpu().numpy())[batch_size-1,:])
                loss_z2 = np.sum(np.abs(cz_pred.detach().cpu().numpy()[batch_size-1,:]*Zernike_alias[3:] - y.detach().cpu().numpy()[batch_size-1,:]))
                loss_zernike = np.minimum(loss_z1, loss_z2)
             # 监督模型进度
                print("Epoch{}:[{}/{} {: .0f}%], Loss:{:.6f} ".format(
                    epoch + 1
                    , samples
                    , epochs*len(batchdata.dataset)
                    , 100*samples/(epochs*len(batchdata.dataset))
                    , loss.data.item()),"Zer loss: ",loss_zernike)
                step11 = epoch+1
                fid.write(' step:'+str(step11)+' ['+str(samples)+'/'+\
                str(epochs*len(batchdata.dataset))+']'+\
                ' loss:'+str(round(loss.cpu().item(),4))+\
                ' loss_zernike:'+str(round(loss_zernike,4)))
            if (batch_idx + 1) % nsnapshot == 0 or batch_idx == (len(batchdata) - 1):
                torch.save(net.state_dict(), './model/'+model_name+'save_step'+repr(epoch + 1)+'.pt')

# 设置随机种子
torch.manual_seed(51)

# 实例化模型
net = Xception()
net = net.to(device)
# 学习率
lr = 0.0001
# 每次小批次训练个数
bs = 16
# 整体数据循环次数
epochs = 50

# 训练模型
fit(net, train_data, lr, epochs)



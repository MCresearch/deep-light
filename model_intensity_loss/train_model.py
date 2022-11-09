import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Zernike import *
from fun import *
import time
import json
from propagation import *

with open("INPUT.json", 'r', encoding='utf-8') as fw:
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

nsnapshot = 100
mm = 8
Phase_option = "ramdom"


class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1 = nn.Conv2d( 1, 32, kernel_size=7, dilation=1, padding='same', dtype=torch.float32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=7, dilation=1, padding='same', dtype=torch.float32)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=7, dilation=1, padding='same', dtype=torch.float32)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=7, dilation=1, padding='same', dtype=torch.float32)
    self.conv5 = nn.Conv2d(256, 64, kernel_size=7, dilation=1, padding='same', dtype=torch.float32)
    self.conv6 = nn.Conv2d(64, 16, kernel_size=7, dilation=1, padding='same', dtype=torch.float32)

    self.mp = nn.MaxPool2d(2)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(8*8*16, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 33)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.1)
        nn.init.normal_(m.bias, 0.0, 0.1)

  def forward(self,x):
    in_size = x.size(0)
    h = self.mp(self.relu(self.conv1(x))) #[48x48]
    h = self.relu(self.conv2(h)) #[48, 48]
    h = self.mp(self.relu(self.conv3(h))) #[24x24]
    h = self.relu(self.conv4(h)) #[24, 24]
    h = self.mp(self.relu(self.conv5(h))) #[12x12]
    h = self.mp(self.relu(self.conv6(h))) #[6x6]
    h = h.view(in_size, -1)
    print("h",np.shape(h))
    h = self.relu(self.fc1(h))
    h = self.relu(self.fc2(h))
    out = self.fc3(h)
    return out



ngrid2 = ngrid//2

## Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#device = torch.device("cpu")
model = Net()
model = model.to(device)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## training
fid = open('train1.log', 'w')

i = 0
# Zer_torch = torch.tensor(Zer[3:, :, :]).to(device)
# mask0_torch = torch.tensor(mask0).to(device)
while 1:
    i += 1
    # prepare minibatch
    #cz = np.random.normal(np.zeros(10), np.exp(-0.2*np.array([1,2,2,2,2,2,3,3,3,3])))
    time_start = time.time()
    init_intens = init_intensity(mm,a0,xx0,mgs)
    Zer,cz = Zer(1,maxZnkOrder,mm,a0,xx0,"ramdom",eeznk,rms,zernike_dir)
    far_field_intens = progagtion(1,mm,a0,xx0,plm,zfh,xxz,init_intens,cz,Zer)
    time_end = time.time()
    img0_img = torch.tensor(np.float32(np.expand_dims(far_field_intens[0,:,:], [0,1]))).to(device) #[1,1,96,96]
    print(np.shape(img0_img))
    # forward pass
    cz_pred = model(img0_img) #[10]
    cz_1 = cz_pred+1-1
    cz_2 = cz_1.detach().cpu().numpy()
    cz_ = np.zeros((1,35))
    cz_[0,2:] = cz_2
    # compute loss
    # Zer,cz = Zer(1,maxZnkOrder,mm,a0,xx0,"confirm",eeznk,rms,zernike_dir)
    far_field_intens_pred = progagtion(1,mm,a0,xx0,plm,zfh,xxz,init_intens,cz_,Zer)
    cz_pred.requires_grad_()
    img1_img = torch.tensor(np.float32(np.expand_dims(far_field_intens_pred[0,:,:], [0,1]))).to(device) #[1,1,96,96]
    loss = torch.sum(torch.abs(img1_img - img0_img))
    #loss = torch.sum(img0_img*torch.log(img1_img))

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step()

    # monitor
    if i%1000 == 0:
        loss_z1 = np.sum(np.abs(cz_pred.detach().cpu().numpy() - cz))
        loss_z2 = np.sum(np.abs(cz_pred.detach().cpu().numpy()*Zernike_alias[3:] - cz))
        loss_zernike = np.minimum(loss_z1, loss_z2)
        print(' step:'+repr(i)+\
            ' loss:'+repr(round(loss.cpu().item(),4))+\
            ' loss_zernike:'+repr(round(loss_zernike.item(),4)), flush = True)
        fid.write(' step:'+repr(i)+\
            ' loss:'+repr(round(loss.cpu().item(),4))+\
            ' loss_zernike:'+repr(round(loss_zernike.item(),4)))

        plt.subplot(2,2,1)
        plt.imshow(np.imag(obj0)[ngrid2//2:ngrid-ngrid2//2, ngrid2//2:ngrid-ngrid2//2], vmin=-1, vmax=1, cmap='seismic')
        plt.subplot(2,2,2)
        plt.imshow(np.imag(obj1.detach().cpu().numpy())[ngrid2//2:ngrid-ngrid2//2, ngrid2//2:ngrid-ngrid2//2], vmin=-1, vmax=1, cmap='seismic')

        plt.subplot(2,2,3)
        plt.imshow( np.abs(img0)**2 + 0.01*np.random.rand(96,96), vmin=0, vmax=0.3, cmap = "jet")
        plt.subplot(2,2,4)
        plt.imshow( np.abs(img1.detach().cpu().numpy())**2, vmin=0, vmax=0.3, cmap = "jet")

        plt.savefig('train.png')
        plt.close()

    if i%(1e5) == 0:
        torch.save(model.state_dict(), 'save_step'+repr(i//100000)+'e5.pt')







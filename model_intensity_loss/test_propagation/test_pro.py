# %%
# import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from Zernike import *
from fun import *
import time
import json
from propagation import *
import torch
import torch.nn as nn
sys.path.append("../")
with open("INPUT_repro.json", 'r', encoding='utf-8') as fw:
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

Zer = Zer1(maxZnkOrder,mm,a0,xx0)
Zer = torch.tensor(Zer).to(device)
init_intens = init_intensity(mm,a0,xx0,mgs)
init_intens = torch.tensor(init_intens).to(device)
cz = np.loadtxt(zernike_dir)
cz = torch.tensor(cz).to(device)
print(cz)

# %%
nor_far = nor_progagtion(nsnapshot,mm,a0,xx0,plm,zfh,xxz,init_intens,cz,Zer)
print(nor_far)

# %%
eg_far_orig = np.loadtxt("./data/256_dl_outIntensity.dat")
eg_nor_far = np.zeros((nsnapshot,ngrid,ngrid))
for i in range(nsnapshot):
    eg_nor_far[i,:,:] = eg_far_orig[i*ngrid:(i+1)*ngrid,:]
    eg_nor_far[i,:,:] = eg_nor_far[i,:,:]/np.max(eg_nor_far[i,:,:])
    


# %%
nor_far = nor_far.cpu().numpy()
diff_outIntensity = np.average(np.abs(eg_nor_far[0,:,:] - np.round(nor_far[0,:,:],1)),axis=0)
print("outIntensity 的差为：")
print(np.linalg.norm(diff_outIntensity,ord=2))


# %%
plt.figure(1, dpi = 300)
plt.contourf(eg_nor_far[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xticks([i*ngrid/4 for i in range(5)],size=10)
plt.yticks([i*ngrid/4 for i in range(5)], size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)
plt.savefig("./eg_norfar.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(nor_far[0,:,:],levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xticks([i*ngrid/4 for i in range(5)],size=10)
plt.yticks([i*ngrid/4 for i in range(5)], size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)
plt.savefig("./norfar.png")
plt.close()

# %% [markdown]
# 



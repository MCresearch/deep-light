# import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
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

mm=8
nsnapshot = 1

time_start = time.time()
init_intens = init_intensity(mm,a0,xx0,mgs)
# Zer,cz = Zer(nsnapshot,maxZnkOrder,mm,a0,xx0,Phase_option,eeznk,rms,zernike_dir)
Zer = Zer1(maxZnkOrder,mm,a0,xx0)
cz = cc(nsnapshot,maxZnkOrder,"random",eeznk,rms,zernike_dir)
far_field_intens = progagtion(nsnapshot,mm,a0,xx0,plm,zfh,xxz,init_intens,cz,Zer)

# c_far_field_intens = np.loadtxt("1108_7_4_dl_down_intensity.dat")
# diff_outIntensity = np.average(np.abs(c_far_field_intens - np.round(far_field_intens[0,:,:],1)),axis=0)
# np.savetxt("./result_propagation/diff_outIntensity",diff_outIntensity)
# print("outIntensity 的差为：")
# print(np.linalg.norm(diff_outIntensity,ord=2) )

# plt.figure(1, dpi = 300)
# plt.contourf(c_far_field_intens)
# plt.savefig("./result_propagation/c_outIntensity.png")
# plt.close()

plt.figure(1, dpi = 300)
plt.contourf(far_field_intens[0,:,:])
plt.savefig("./result_propagation/p_far_field_intens.png")
plt.close()

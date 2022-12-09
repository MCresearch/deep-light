# import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Zernike import *
import time
import json
from propagation import *
import numba
from numba import jit

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

nsnapshot = 1000
mm = 8
gy1,gx1 = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1,ngrid-n1,ngrid),dxyz*np.linspace(1-n1,ngrid-n1,ngrid))
time_start = time.time()
init_intens = init_intensity(mm,a0,xx0,mgs,gx1,gy1)
time_end = time.time()
print("tmie1",time_end-time_start)

Zer,cz = Zer(nsnapshot,maxZnkOrder,mm,a0,xx0,Phase_option,eeznk,rms,zernike_dir,gx1,gy1)
time_end1 = time.time()
print("tmie2",time_end1-time_end)

far_field_intens = progagtion(nsnapshot,mm,a0,xx0,plm,zfh,xxz,init_intens,cz,Zer,gx1,gy1,gx2,gy2)
time_end2 = time.time()
print("tmie3",time_end2-time_end1)
far_field_intens = np.float32(far_field_intens)
np.save("./1115test.npy",far_field_intens)
time_end3 = time.time()
print("tmie",time_end3-time_end2)
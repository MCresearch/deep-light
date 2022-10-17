import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

m_grid = 128
rms = 1
aaz = 20
nsnapshots = 1000
dir = "/home/xianyuer/data/35_xception_200000_rms1/test/"

real_out_all_1 = np.loadtxt(dir+"dl_down_intensity.dat")
real_out_all = np.zeros((nsnapshots,128,128))
real_out_all_nor = np.zeros((nsnapshots,128,128))
for i in range(nsnapshots):
    real_out_all[i,:,:]=real_out_all_1[i*128:(i+1)*128,:]
    real_out_all_nor[i,:,:] = real_out_all[i,:,:]/np.max(real_out_all[i,:,:])
np.save(dir+"nor_outintensity.npy",real_out_all_nor)
np.save(dir+"outintensity.npy",real_out_all)

zernike = np.loadtxt(dir+"dl_zernike_coeff.dat")
np.save(dir+"zernike.npy",zernike)


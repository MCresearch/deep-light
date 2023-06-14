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

c_outintensity_0 = np.loadtxt("./dl_outIntensity.dat")
p_outintensity = np.load("./result_propagation/sumnor_outintensity.npy")

c_outintensity = np.zeros((2,8,8))
for i in range(2):
    c_outintensity[i,:,:] = c_outintensity_0[i*8:(i+1)*8,:]
    sum1 = np.sum(c_outintensity[i,:,:])
    c_outintensity[i,:,:] = c_outintensity[i,:,:]/sum1
    plt.figure(1, dpi = 300)
    plt.contourf(c_outintensity[i,:,:],levels=100, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig("./result_propagation/c_outintensity_"+str(i)+".png")
    plt.close()
    
    plt.figure(1, dpi = 300)
    plt.contourf(p_outintensity[i,:,:],levels=100, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig("./result_propagation/p_outintensity_"+str(i)+".png")
    plt.close()

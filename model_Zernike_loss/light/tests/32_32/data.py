
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nsnapshots = 10000 # 指定帧数
distribution = "0.5"
nzernike = 35
n_grid = 32

intensity = np.zeros((nsnapshots,n_grid,n_grid))
zernike = np.zeros((nsnapshots,nzernike))

intensity_dir = "./dl_outIntensity.dat"
#zernike_dir = "./dl_zernike_coeff.dat"

x = np.loadtxt(intensity_dir)
#zernike = np.loadtxt(zernike_dir)

for i in range(nsnapshots):
    intensity[i,:,:] = x[n_grid*i:n_grid*(i+1),]

print("intensity",intensity[1,:,:])
#print("zernike",zernike)
print("max of intensity = ", np.max(intensity))

#print("nsnapshot = %s" % nsnapshot)
print("intensity shape = ", np.shape(intensity))
#print("zernike shape = ", np.shape(zernike))

np.save('../../../machinelearning/0620/data/outIntensity_%d_%s_%d_%d'%(nzernike,distribution,n_grid,nsnapshots),intensity)
#np.save('dl_zernike_coeff_104_10000.npy', zernike)




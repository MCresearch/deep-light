
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nsnapshots = 1000 # 指定帧数
distribution = "0.5"
nzernike = 65
n_grid = 128

intensity = np.zeros((nsnapshots,n_grid,n_grid))
#zernike = np.zeros((nsnapshots,nzernike))

intensity_dir = "../../../data/intensity_test_diff.dat"
#zernike_dir = "../../../data/raw_zernike_coeff.dat"

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

#np.save('../../../data/raw_outIntensity_%d_%d_%d'%(n_grid,nzernike,nsnapshots),intensity)
#np.save('../../../data/raw_zernike_coeff_%d__%d_%d'%(n_grid,nzernike,nsnapshots), zernike)
np.save('../../../data/intensity_test_diff', intensity)



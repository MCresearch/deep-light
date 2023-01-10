
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nsnapshots = 10000 # 指定帧数
distribution = "0.5"
nzernike = 35

intensity = np.zeros((nsnapshots,64,64))
zernike = np.zeros((nsnapshots,nzernike))

intensity_dir = "./64_64/dl_outIntensity.dat"
#zernike_dir = "./dl_zernike_coeff.dat"

x = np.loadtxt(intensity_dir)
#zernike = np.loadtxt(zernike_dir)

for i in range(nsnapshots):
    intensity[i,:,:] = x[64*i:64*(i+1),]

print("intensity",intensity[1,:,:])
#print("zernike",zernike)
print("max of intensity = ", np.max(intensity))

#print("nsnapshot = %s" % nsnapshot)
print("intensity shape = ", np.shape(intensity))
#print("zernike shape = ", np.shape(zernike))

np.save('/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/0620/data/outIntensity_%d_%s_64_%d'%(nzernike,distribution,nsnapshots),intensity)
#np.save('dl_zernike_coeff_104_10000.npy', zernike)




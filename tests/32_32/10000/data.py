import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######### parameters setting ########
data_size = [100, 1000, 10000, 17000, 43000, 100000]
model_name = "0511_1"
epoch = [8, 16, 600, 500, 400, 300]
batch_size=16
seed = 12333345
data_time = "20220511_104_0.1c"
##data_time = "202103_10_0.1c"
##data_time = "20210305_mix"
input_model = False
#model_path = "0921_2_20210322_10_0.1_17000_300.h5"
#####################################

intensity = np.zeros((10000,32,32))
zernike = np.zeros((10000,104))

intensity_dir = "dl_outIntensity.dat"
zernike_dir = "dl_zernike_coeff.dat"

x = np.loadtxt(intensity_dir)
zernike = np.loadtxt(zernike_dir)

for i in range(10000):
    intensity[i,:,:] = x[32*i:32*(i+1),]

print("intensity",intensity)
print("zernike",zernike)
print("max of intensity = ", np.max(intensity))

#print("nsnapshot = %s" % nsnapshot)
print("intensity shape = ", np.shape(intensity))
print("zernike shape = ", np.shape(zernike))

np.save('dl_outIntensity.npy', intensity)
np.save('dl_zernike_coeff.npy', zernike)

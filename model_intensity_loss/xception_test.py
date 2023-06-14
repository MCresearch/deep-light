
######### import session #############
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras.backend as K
#K.set_image_dim_ordering('tf')
import sys
sys.path.append("../")
import time
from keras import losses
from keras import Input
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from tensorflow.keras.layers import Multiply ##
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.callbacks
import numpy as np
import random
from tensorflow.keras.models import load_model
import keras.initializers
import matplotlib.pyplot as plt
import json
from NpEncoder import *
from func import *
from keras_applications import xception
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import tensorflow as tf
from xception import Xception
######################################
time_start=time.time()
# rms = 0.5
nsnapshot = 4000
dir = "/home/xianyuer/data/paper1/model0/noise0.01/"
model_dir = "/home/xianyuer/data/35_rms4_aaz36_200000/35_128_200000_50-100_midloop4_220718_200000_50.h5"
# intensity_dir1 = "/home/xianyuer/data/35_rms4_aaz36_200000/rms1_1000_36_nor_outIntensity_35_0-1_1_1000.npy"
# intensity_dir2 = "/home/xianyuer/data/35_rms4_aaz36_200000/rms2_1000_36_nor_outIntensity_35_0-1_2_1000.npy"
# intensity_dir3 = "/home/xianyuer/data/35_rms4_aaz36_200000/rms3_1000_36_nor_outIntensity_35_0-1_3_1000.npy"
# intensity_dir4 = "/home/xianyuer/data/35_rms4_aaz36_200000/5_2000nor_outIntensity_65_0-1_4_2000.npy"
# intensity_dir1 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms1/rms1_new_1000_noise_out_0.0010.npy"
# intensity_dir2 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms2/rms2_new_1000_noise_out_0.0010.npy"
# intensity_dir3 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms3/rms3_new_1000_noise_out_0.0010.npy"
# intensity_dir4 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms4/rms4_new_1000_noise_out_0.0010.npy"
intensity_dir1 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms1/rms1_new_1000_noise_out_0.0100.npy"
intensity_dir2 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms2/rms2_new_1000_noise_out_0.0100.npy"
intensity_dir3 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms3/rms3_new_1000_noise_out_0.0100.npy"
intensity_dir4 = "/home/xianyuer/data/paper1/model_rmsmix/testnoise/paper_noise_rms4/rms4_new_1000_noise_out_0.0100.npy"
zernike_dir1 ="/home/xianyuer/data/35_rms4_aaz36_200000/rms1_1000_36_zernike_35_0-1_1_1000.npy"
zernike_dir2 ="/home/xianyuer/data/35_rms4_aaz36_200000/rms2_1000_36_zernike_35_0-1_2_1000.npy"
zernike_dir3 ="/home/xianyuer/data/35_rms4_aaz36_200000/rms3_1000_36_zernike_35_0-1_3_1000.npy"
zernike_dir4 ="/home/xianyuer/data/35_rms4_aaz36_200000/5_2000zernike_65_0-1_4_2000.npy"

test_file_name = dir+"zernike_test_real.txt"
test_file_name_p = dir+"zernike_test_predict.txt"
test_file_name_diff = dir+"zernike_test_diff.txt"

Zernike_alias = np.array([-1] * 3 + [1] * 4 + [-1] * 5 + [1] * 6 + [-1] * 7 + [1] * 8, dtype=np.float32)
x1 = np.load(intensity_dir1)
x2 = np.load(intensity_dir2)
x3 = np.load(intensity_dir3)
x4 = np.load(intensity_dir4)
y1 = np.load(zernike_dir1)
y2 = np.load(zernike_dir2)
y3 = np.load(zernike_dir3)
y4 = np.load(zernike_dir4)

y = np.concatenate((y1,y2))
y = np.concatenate((y,y3))
y = np.concatenate((y,y4))
x = np.concatenate((x1,x2))
x = np.concatenate((x,x3))
x = np.concatenate((x,x4))
y = y[:,2:]

if os.path.exists(dir):
    print("已存在")
else:
    os.mkdir(dir)
    


model = Xception(input_shape = (128, 128, 1),
                 pooling = 'avg',
                 backend=keras.backend,
                 layers=keras.layers,
                 models=keras.models,
                 utils=keras.utils,
                 middle_loop=4, #renxi added
                 outdim = 33,
        )
model.load_weights(model_dir)
############ data and epoch specification #################
print("max of x = ", np.max(x))
print("nsnapshot = %s" % nsnapshot)
print("x shape = ", np.shape(x))
print("y shape = ", np.shape(y))

test_x = x[:nsnapshot].reshape([nsnapshot, 128, 128, 1])
test_y = y[:nsnapshot]    

print("test_x shape = ", np.shape(test_x))
print("test_y shape = ", np.shape(test_y))

test_yp= model.predict(test_x)
time_end = time.time()
print("time cost: ", time_end-time_start, "s")
test_ypmin = np.zeros((nsnapshot,33))
# for i in range(nsnapshot):
#     loss_z1 = np.mean(pow(test_yp[i,:] - test_y[i,:],2))
#     loss_z2 = np.mean(pow(test_yp[i,:]*Zernike_alias - test_y[i,:],2))
#     if loss_z1 <= loss_z2:
#         test_ypmin[i,:] = test_yp[i,:]
#     else:
#         test_ypmin[i,:] = test_yp[i,:]*Zernike_alias
test_ypmin =  test_yp       

for i in range(4):
    print("rms",i+1,":",np.sqrt(np.mean(np.power(test_yp[i*1000:(i+1)*1000,:]-test_y[i*1000:(i+1)*1000,:], 2))))
    print("rms_min",i+1,":",np.sqrt(np.mean(np.power(test_ypmin[i*1000:(i+1)*1000,:]-test_y[i*1000:(i+1)*1000,:], 2))))

np.savetxt(test_file_name, test_y)
np.savetxt(test_file_name_p, test_ypmin)
np.savetxt(test_file_name_diff, test_y-test_ypmin)

# if not os.path.exists(fig_dir):
#     os.mkdir(fig_dir)

# for i in range(1):
#     plt.figure(1, dpi = 400)
#     plt.bar(np.array(range(63)),test_y[i*10], label = "real")
#     plt.bar(np.array(range(63)),test_yp[i*10], label = "predict")
#     plt.xticks(range(1,66,4))
#     plt.xlabel("Zernike order")
#     plt.ylabel("Zernike coefficient values")
#     plt.title("Test set No.%d, model = xceptionfull_65_4" % (i*10))
#     plt.legend()
#     plt.savefig(os.path.join(fig_dir, str(i*10)+"_test_xceptionfull_65_4.png"))
#     plt.close()

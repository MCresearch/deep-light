
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
rms = 4
num = 2
aaz = 36
nsnapshot = 3
model_dir = "/home/xianyuer/data/35_defocus/35_128_1-150_midloop4_defocus_220908_200000_150.h5"
model = Xception(input_shape = (128, 128, 2),
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
##
intensity_dir = "/home/xianyuer/data/35_rms4_aaz36_200000/5_2000nor_outIntensity_65_0-1_4_2000.npy"
zernike_dir ="/home/xianyuer/data/35_rms4_aaz36_200000/5_2000zernike_65_0-1_4_2000.npy"
de_dir = "/home/xianyuer/data/35_defocus/5_2000_de_1.45nor_outintensity.npy"
x = np.load(intensity_dir)
y = np.load(zernike_dir)
y = y[:,2:]
xde = np.load(de_dir)

xx1 = x[:1000,:,:].reshape([1000,128, 128])
xxde1 = xde[:1000,:,:].reshape([1000,128, 128])
test_x = np.zeros([1000,128,128,2])
test_x[:, :, :, 0] = xx1 
test_x[:, :, :, 1] = xxde1 
test_y = y[:1000]
print("test_x shape = ", np.shape(test_x))
print("test_y shape = ", np.shape(test_y))

dir = "/home/xianyuer/data/35_defocus/35_128_1-150_midloop4_defocus/"
if not os.path.exists(dir):
    os.mkdir(dir)

time_start=time.time()
test_yp= model.predict(test_x)
time_end = time.time()
print("time cost: ", time_end-time_start, "s")
with open(dir+"ems.txt", encoding="utf-8",mode="a") as file:  
    file.write(str(np.mean(np.power(test_yp-test_y, 2))))
    file.write("\n")

test_file_name = dir+"zernike_test_real.txt"
test_file_name_p = dir+"zernike_test_predict.txt"
test_file_name_diff = dir+"zernike_test_diff.txt"

np.savetxt(test_file_name, test_y)
np.savetxt(test_file_name_p, test_yp)
np.savetxt(test_file_name_diff, test_y-test_yp)

   
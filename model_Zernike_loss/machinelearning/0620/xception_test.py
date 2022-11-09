
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
import json
with open("INPUT.json", 'r', encoding='utf-8') as fw:
    injson = json.load(fw)

rms = np.pi*injson['test']['rms']
nsnapshot = injson['test']['nsnapshot']
model_dir = injson['test']['model_dir'] # model dir
intensity_dir = injson['test']['intensity_dir'] # Far-field light intensity path
zernike_dir =injson['test']['zernike_dir'] # Zernike coefficients path
dir = injson['test']['dir'] # File saving path
'''
rms = 1
nsnapshot = 1000
model_dir = "/home/xianyuer/data/35_xception_200000_rms1/35_128_1-100_midloop4_221007_200000_150.h5"
intensity_dir = "/home/xianyuer/data/35_xception_200000_rms1/data/5_nor_outintensity.npy"
zernike_dir ="/home/xianyuer/data/35_xception_200000_rms1/data/5_zernike_35.npy"
'''
model = Xception(input_shape = (128, 128,1),
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


x = np.load(intensity_dir)
y = np.load(zernike_dir)
y = y[:,2:]

print("max of x = ", np.max(x))
print("nsnapshot = %s" % nsnapshot)
print("x shape = ", np.shape(x))
print("y shape = ", np.shape(y))

test_x = x[:nsnapshot,:,:].reshape([nsnapshot, 128, 128, 1])
test_y = y[:nsnapshot]    
print("test_x shape = ", np.shape(test_x))
print("test_y shape = ", np.shape(test_y))
time_start=time.time()
test_yp= model.predict(test_x)
time_end = time.time()
print("time cost: ", time_end-time_start, "s")


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

   
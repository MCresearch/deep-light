
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
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau
import keras.callbacks
import numpy as np
import random
from keras.models import load_model
import keras.initializers
import matplotlib.pyplot as plt
import json
from NpEncoder import *
from func import *
from keras_applications import xception
import keras
from tensorflow.keras.utils import plot_model
from keras.models import Model

######################################

############# data and epoch specification #################
###

intensity_dir = "/home/xianyuer/yuer/num_mechinelearning/tests/32_32/dl_outIntensity.npy"
zernike_dir = "/home/xianyuer/yuer/num_mechinelearning/tests/32_32/dl_zernike_coeff.npy"
x = np.load(intensity_dir)
y = np.load(zernike_dir)

###
'''
# 计算区域 = 0.55 m
intensity_dir2 = "/home/lrx/work/10_MO2020/202103_10_0.1c/0.55/intensity_202103c_last2000_10.npy"

# 口径 = 0.151 m
intensity_dir2 = "/home/lrx/work/10_MO2020/202103_10_0.1c/radius0.151/intensity_radius0.151.npy"
'''
model_dir = "/home/xianyuer/yuer/num_mechinelearning/mechinelearning/10Zernike_test/0511_1_20220511_104_0.1c_10000_20.h5"
fig_dir = "./fig"
nsnapshot = 1000

###
'''
intensity_dir = "/home/xianyuer/yuer/num/tests/128_128/dl_outIntensity_0.239100.dat"
zernike_dir = "/home/xianyuer/yuer/num/tests/128_128/dl_zernike_coeff_0.239100.dat"
x = np.loadtxt(intensity_dir,comments='#')
y = np.loadtxt(zernike_dir,comments='#')
y = y[:10,1]
x= np.expand_dims(x, axis=0)
y = np.expand_dims(y, axis=0)
'''
###

print("max of x = ", np.max(x))
print("nsnapshot = %s" % nsnapshot)
print("x shape = ", np.shape(x))
print("y shape = ", np.shape(y))

train_x = x[:nsnapshot].reshape([nsnapshot, 32, 32, 1])
train_y = y[:nsnapshot]

test_x = x[-1:].reshape([1, 32, 32, 1])
test_y = y[-1:]    
print("train_x shape = ", np.shape(train_x))
print("train_y shape = ", np.shape(train_y))
print("test_x shape = ", np.shape(test_x))
print("test_y shape = ", np.shape(test_y))
#test_x2 = np.load(intensity_dir2)[-1000:].reshape([1000, 128, 128, 1])

model = load_model(model_dir)
    
train_yp = model.predict(train_x)
test_yp = model.predict(test_x)
#test_yp2 = model.predict(test_x2)
print(np.sqrt(np.mean(np.power(test_yp-test_y, 2))))
#print(np.sqrt(np.mean(np.power(test_yp2-test_y, 2))))

'''
test_out = ""
test_out_p = ""
#test_out_p2 = ""
for isnapshot in range(1000):
    for i in range(103):
        test_out += str(test_y[isnapshot, i]) + " "
        test_out_p += str(test_yp[isnapshot, i]) + " "
        #test_out_p2 += str(test_yp2[isnapshot, i]) + " "
    test_out += "\n"
    test_out_p += "\n"
    #test_out_p2 += "\n"
'''
test_file_name = "/home/xianyuer/yuer/num_mechinelearning/mechinelearning/10Zernike_test/zernike_test_real.txt"
test_file_name_p = "/home/xianyuer/yuer/num_mechinelearning/mechinelearning/10Zernike_test/zernike_test_predict.txt"
#test_file_name_p2 = "zernike_test_predict2.txt"
np.savetxt(test_file_name, test_y)
np.savetxt(test_file_name_p, test_yp)
print(test_y-test_yp)




if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

for i in range(10):
    plt.figure(1, dpi = 400)
    plt.plot(train_y[i*100], label = "real")
    plt.plot(train_yp[i*100], label = "predict")
    plt.xticks([i for i in range(10)], [i for i in range(1, 11)])
    plt.xlabel("Zernike order")
    plt.title("Train set No.%d, model = 0919" % (i*100))
    plt.legend()
    plt.savefig(os.path.join(fig_dir, str(i*100)+"_train_0919.png"))
    plt.close()
   
    plt.figure(1, dpi = 400)
    plt.plot(test_y[i*100], label = "real")
    plt.plot(test_yp[i*100], label = "predict")
    plt.xticks([i for i in range(10)], [i for i in range(1, 11)])
    plt.xlabel("Zernike order")
    plt.title("Test set No.%d, model = 0919" % (i*100))
    plt.legend()
    plt.savefig(os.path.join(fig_dir, str(i*100)+"_test_0919.png"))
    plt.close()

   
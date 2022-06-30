
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

# intensity_gauss_dir = "/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/zernike_order/dl_outIntensity_9_10000.npy"
# zernike_gauss_dir = "/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/zernike_order/dl_zernike_coeff_9_10000.npy"
intensity_dir = "/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/0620/data/outIntensity_35_2_64_10000.npy"
zernike_dir = "/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/0620/data/zernike_220620_2_35_10000.npy"
# intensity_dir = "/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/distribution/dl_inIntensity_9_0.1_64_10000.npy"
# zernike_dir = "/home/xianyuer/yuer/21zishiying/21zishiying/0.1均匀/zernike_20220617_9.npy"

# x1 = np.load(intensity_gauss_dir)
# y1 = np.load(zernike_gauss_dir)

x2 = np.load(intensity_dir)
y2 = np.load(zernike_dir)
# x3 = np.vstack((x1[:1],x2[:1]))
# y3 = np.vstack((y1[:1],y2[:1]))
###
'''
# 计算区域 = 0.55 m
intensity_dir2 = "/home/lrx/work/10_MO2020/202103_10_0.1c/0.55/intensity_202103c_last2000_10.npy"

# 口径 = 0.151 m
intensity_dir2 = "/home/lrx/work/10_MO2020/202103_10_0.1c/radius0.151/intensity_radius0.151.npy"
'''
model_dir = "/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/0620/xceptionfull_35_2_64_220620_10000_50.h5"
fig_dir = "./fig"
nsnapshot = 10000

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

# print("max of x1 = ", np.max(x1))
# print("nsnapshot = %s" % nsnapshot)
# print("x1 shape = ", np.shape(x1))
# print("y1 shape = ", np.shape(y1))

# train_x1 = x1[:nsnapshot].reshape([nsnapshot, 64, 64, 1])
# train_y1 = y1[:nsnapshot]

# test_x1 = x1[:100:].reshape([100, 64, 64, 1])
# test_y1 = y1[:100:]    

train_x2 = x2[:nsnapshot].reshape([nsnapshot, 64, 64, 1])
train_y2 = y2[:nsnapshot]
test_x2 = x2[:100].reshape([100, 64, 64, 1])
test_y2 = y2[:100]  
# print("train_x1 shape = ", np.shape(train_x1))
# print("train_y1 shape = ", np.shape(train_y1))
# print("test_x1 shape = ", np.shape(test_x1))
# print("test_y1 shape = ", np.shape(test_y1))

print("train_x2 shape = ", np.shape(train_x2))
print("train_y2 shape = ", np.shape(train_y2))
print("test_x2 shape = ", np.shape(test_x2))
print("test_y2 shape = ", np.shape(test_y2))
#test_x2 = np.load(intensity_dir2)[-1000:].reshape([1000, 128, 128, 1])

# print("test_x3 shape = ", np.shape(x3))
# print("test_y3 shape = ", np.shape(y3))

model = load_model(model_dir)
    
# train_yp1 = model.predict(train_x1)
# test_yp1 = model.predict(test_x1)
train_yp2 = model.predict(train_x2)
test_yp2= model.predict(test_x2)
# test_yp3= model.predict(x3)
#test_yp2 = model.predict(test_x2)
# print("111")
# print(np.sqrt(np.mean(np.power(test_yp1-test_y1, 2))))
print("222")
print(np.sqrt(np.mean(np.power(test_yp2-test_y2, 2))))
# print("333")
# print(np.sqrt(np.mean(np.power(test_yp3-y3, 2))))
#print(np.sqrt(np.mean(np.power(test_yp2-test_y, 2))))


# test_out = ""
# test_out_p = ""
# #test_out_p2 = ""
# for isnapshot in range(100):
#     for i in range(35):
#         test_out += str(test_y2[isnapshot, i]) + " "
#         test_out_p += str(test_yp2[isnapshot, i]) + " "
#         #test_out_p2 += str(test_yp2[isnapshot, i]) + " "
#     test_out += "\n"
#     test_out_p += "\n"
#     #test_out_p2 += "\n"

test_file_name = "./zernike_test_real_35_10000_64.txt"
test_file_name_p = "./zernike_test_predict_35_10000_64.txt"
#test_file_name_p2 = "zernike_test_predict2.txt"
np.savetxt(test_file_name, test_y2)
np.savetxt(test_file_name_p, test_yp2)
# print(test_y-test_yp)




if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

for i in range(10):
    plt.figure(1, dpi = 400)
    plt.plot(train_y2[i*1000], label = "real")
    plt.plot(train_yp2[i*1000], label = "predict")
    #plt.xticks([i for i in range(10)], [i for i in range(1, 105)])
    plt.xlabel("Zernike order")
    plt.ylabel("Zernike coefficient values")
    plt.title("Train set No.%d, model = xception_35_2" % (i*1000))
    plt.legend()
    plt.savefig(os.path.join(fig_dir, str(i*1000)+"_train_xception_35_2.png"))
    plt.close()
   
    plt.figure(1, dpi = 400)
    plt.plot(test_y2[i*10], label = "real")
    plt.plot(test_yp2[i*10], label = "predict")
    plt.xticks(range(35))
    plt.xlabel("Zernike order")
    plt.ylabel("Zernike coefficient values")
    plt.title("Test set No.%d, model = xception_35_2" % (i*10))
    plt.legend()
    plt.savefig(os.path.join(fig_dir, str(i*10)+"_test_xception_35_2.png"))
    plt.close()

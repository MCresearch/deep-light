######### parameters setting ########
data_size = [200000, 10000, 10000, 17000, 43000, 100000]
model_name = "35_128_1-100_midloop4_intloss"
epoch = [100, 100,600, 500, 400, 300]
batch_size=20
#batch_size=8
seed = 12333345
data_time = "221129"
input_model = False
dir = "/data/home/scv1925/run/3_zhangxianyue/deep-light/35_rms4_200000/"
# input_model = "./35_128_1-50_midloop4_loss_noise_0.05_220823_200000_50.h5"
#model_path = "0921_2_20210322_10_0.1_17000_300.h5"
#####################################


######### import session #############
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras.backend as K
#K.set_image_dim_ordering('tf')
import sys
sys.path.append("../")
import time
from tensorflow.keras import losses
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.callbacks
import numpy as np
import random
from tensorflow.keras.models import load_model
import tensorflow.keras.initializers
import matplotlib.pyplot as plt
import json
from NpEncoder import *
from func import *
from keras_applications import xception
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.callbacks import Callback
from xception import Xception
import tensorflow.keras.backend as K

###################################
from Zernike import *
from fun import *
from propagation import *

with open("INPUT.json", 'r', encoding='utf-8') as fw:
    injson = json.load(fw)

mm = injson['data']['mm'] 
mgs = injson['data']['mgs']
a0 = injson['data']['a0']
xx0 = injson['data']['xx0']
plm = injson['data']['plm']
zfh = injson['data']['zfh']
xxz = injson['data']['xxz']
minZnkDim = injson['data']['minZnkDim']
maxZnkOrder = injson['data']['maxZnkOrder']
rms = injson['data']['rms']
eeznk = injson['data']['eeznk']
dir = injson['data']['dir']
Phase_option = injson['data']['Phase_option']  ##"random" "confirm" 
nsnapshot = injson['data']['nsnapshot'] 
zernike_dir = injson['data']['zernike_dir'] 
nxzz = a0*xx0
ngrid = pow(2,mm)
n1 = ngrid/2 + 1
aa0 = xx0*a0
dxy0 = aa0/ngrid
airy = 1.22*plm*zfh/(2*a0)
aaz = airy*xxz
dxyz = aaz/ngrid
ngrid2 = ngrid//2
a02 = a0*a0

nsnapshot = 5000
mm = 8
Phase_option = "ramdom"

Zernike_alias = np.array([-1] 
                            + [1] * 2
                            + [-1] * 3
                            + [1] * 4
                            + [-1] * 5
                            + [1] * 6
                            + [-1] * 7
                            + [1] * 8, dtype=np.float32)
Zer = Zer1(maxZnkOrder,mm,a0,xx0)
init_intens = init_intensity(mm,a0,xx0,mgs)
###################################
def my_loss(y_true, y_pred):
    
    far_field_intens_pred = progagtion1(1,mm,a0,xx0,plm,zfh,xxz,init_intens,y_pred,Zer)
    img1_img = torch.tensor(np.float32(np.expand_dims(far_field_intens_pred[0,:,:], [0,1]))).to(device) #[1,1,96,96]
    loss = torch.sum(torch.abs(img1_img - img0_img))
    loss.requires_grad_(True) 

    nzernike = y_pred.shape[-1]
    sym = np.ones(nzernike, dtype=np.float32)
    iorder = 2
    lower = 0
    upper = 2
    while lower<nzernike:
        upper = min([upper, nzernike-1])
        sym[lower:upper+1] = -1
        iorder += 2
        lower += 2*iorder-1
        upper += 2*iorder+1
    sym = tf.convert_to_tensor(sym)
    loss = tf.convert_to_tensor(0.0)

    for iss in range(batch_size):
        loss_1 = tf.reduce_mean(tf.square(y_pred[iss] - y_true[iss]), axis=-1)
        loss_2 = tf.reduce_mean(tf.square(y_pred[iss] - tf.multiply(y_true[iss], sym)), axis=-1)
        loss = tf.add(loss, tf.minimum(loss_1, loss_2))
    return tf.divide(loss, tf.convert_to_tensor(float(batch_size)))
        
            

############# data and epoch specification #################
    
intensity_dir = dir+"1_50000nor_outIntensity_65_0-1_4_50000.npy"
zernike_dir = dir+"1_50000zernike_65_0-1_4_50000.npy"

intensity_dir2 =dir+"2_50000nor_outIntensity_65_0-1_4_50000.npy"
zernike_dir2 =dir+"2_50000zernike_65_0-1_4_50000.npy"

intensity_dir3 =dir+"3_50000nor_outIntensity_65_0-1_4_50000.npy"
zernike_dir3 =dir+"3_50000zernike_65_0-1_4_50000.npy"

intensity_dir4 =dir+"4_50000nor_outIntensity_65_0-1_4_50000.npy"
zernike_dir4 =dir+"4_50000zernike_65_0-1_4_50000.npy"

intensity_dir5 =dir+"5_2000nor_outIntensity_65_0-1_4_2000.npy"
zernike_dir5 =dir+"5_2000zernike_65_0-1_4_2000.npy"

for i in range(0, 1):
    nsnapshot = data_size[i]
    nepoch = epoch[i]
    x1 = np.load(intensity_dir)
    y1 = np.load(zernike_dir)

    x2 = np.load(intensity_dir2)
    y2 = np.load(zernike_dir2)

    x3 = np.load(intensity_dir3)
    y3 = np.load(zernike_dir3)

    x4 = np.load(intensity_dir4)
    y4 = np.load(zernike_dir4)

    x5 = np.load(intensity_dir5)
    y5 = np.load(zernike_dir5)

    y = np.concatenate((y1,y2))
    y = np.concatenate((y,y3))
    y = np.concatenate((y,y4))
    y = np.concatenate((y,y5))
    y = y[:,2:]

    x = np.concatenate((x1,x2))
    x = np.concatenate((x,x3))
    x = np.concatenate((x,x4))
    x = np.concatenate((x,x5))

    print("nsnapshot = %s" % nsnapshot)
    print("x shape = ", np.shape(x))
    print("y shape = ", np.shape(y))
    train_x = x[:nsnapshot].reshape([nsnapshot,128, 128, 1])
    train_y = y[:nsnapshot]

    test_x = x[-1000:].reshape([1000, 128, 128, 1])
    test_y = y[-1000:]
    print("test_x shape = ", np.shape(test_x))
    print("test_y shape = ", np.shape(test_y))
    
    ############ model specification ##############
    try:
        if(input_model == False):
            model = Xception(input_shape = (128,  128, 1),
                 pooling = 'avg',
                 backend=keras.backend,
                 layers=keras.layers,
                 models=keras.models,
                 utils=keras.utils,
                 middle_loop=4, #renxi added
                 outdim = 33,
        )
        else:
            model = Xception(input_shape = (128, 128, 1),
                 pooling = 'avg',
                 backend=keras.backend,
                 layers=keras.layers,
                 models=keras.models,
                 utils=keras.utils,
                 middle_loop=4, #renxi added
                 outdim = 33,)
            model.load_weights(input_model)
            print("Model loaded from " + input_model)
        model.compile(loss=my_loss,
                    optimizer='adam') #mean_squared_error为损失函数，adam优化器
        model_callback = keras.callbacks.Callback() #回调函数
        
        file_path = './weights-improvement-{epoch:03d}.h5'
        modelcheckpoint = ModelCheckpoint(filepath = file_path, monitor='val_loss',save_best_only=True, mode='min',period = 20)
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=5, min_lr = 1e-8) #当标准评估停止提升时，降低学习速率。
        #batch_print_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch,logs: print(model.predict(train_x)))
        history = History() #把所有事件都记录到 History 对象的回调函数。这个回调函数被自动启用到每一个 Keras 模型。History 对象会被模型的 fit 方法返回。
        callbacks = [modelcheckpoint, reduceLR, history]
        
    except ValueError:
        print("Model specification error.")
    ############ End of model specification ##############

    ############ Model fitting #################
    time_start=time.time()
    print("data size = %s" % nsnapshot)
    print("Epochs = %s" % nepoch)
    model.fit(x = train_x, 
            y = train_y, 
            epochs = nepoch, 
            batch_size = batch_size, 
            validation_data = (test_x, test_y),
            shuffle = True,
            verbose = 2,
            callbacks=callbacks)
    time_end = time.time()
    print("time cost: ", time_end-time_start, "s")

    ############ End model fitting #################

    model.save_weights(model_name+"_"+data_time+"_"+str(nsnapshot)+"_"+str(nepoch)+".h5")
    with open(model_name+"_"+data_time+"_"+str(nsnapshot)+"_"+str(nepoch)+"_history.json", 'w') as f:
        json.dump(history.history, f, cls = NpEncoder)


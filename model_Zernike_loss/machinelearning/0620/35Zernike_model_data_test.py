######### parameters setting ########
'''
data_size = [200000, 10000, 30000, 100000]
model_name = "35_128_1-100_midloop4"
epoch = [150, 150, 150, 150]
batch_size=16
#batch_size=8
seed = 12333345
data_time = "221007"
input_model = False
dir = "/data/home/scv1925/run/3_zhangxianyue/deep-light/35_rms1_200000/"
#input_model = "./35_128_200000_50-100_midloop4_220718_200000_50.h5"
#model_path = "0921_2_20210322_10_0.1_17000_300.h5"
    
intensity_dir = dir+"data/1_nor_outintensity.npy"
zernike_dir = dir+"data/1_zernike_35.npy"
'''
#####################################
import json
with open("INPUT.json", 'r', encoding='utf-8') as fw:
    injson = json.load(fw)

data_size = injson['train']['data_size'] # snapshot
model_name = injson['train']['model_name'] 
epoch = injson['train']['epoch']
batch_size = injson['train']['batch_size']
seed = injson['train']['seed']
data_time = injson['train']['data_time']
input_model = injson['train']['input_model'] # False or model dir
dir = injson['train']['dir'] # File saving path
intensity_dir = injson['train']['intensity_dir'] # Far-field light intensity path
zernike_dir = injson['train']['zernike_dir'] # Zernike coefficients path


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

############# data and epoch specification #################

x = np.load(intensity_dir)
y = np.load(zernike_dir)

y = y[:,2:]

print("x shape = ", np.shape(x))
print("y shape = ", np.shape(y))

for i in range(1):
    nsnapshot = data_size[i]
    nepoch = epoch[i]
    print("nsnapshot = %s" % nsnapshot)
 
    train_x = x[:nsnapshot].reshape([nsnapshot,128, 128, 1])
    train_y = y[:nsnapshot]

    test_x = x[-1000:].reshape([1000, 128, 128, 1])
    test_y = y[-1000:]
    print("test_x shape = ", np.shape(test_x))
    print("test_y shape = ", np.shape(test_y))
    
    ############ model specification ##############
    try:
        if(input_model == "False"):
            model = Xception(input_shape = (128, 128, 1),
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
        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam') #mean_squared_error为损失函数，adam优化器
        model_callback = keras.callbacks.Callback() #回调函数
        
        file_path = './weights-improvement-{epoch:03d}.h5'
        modelcheckpoint = ModelCheckpoint(filepath = file_path, monitor='val_loss',save_best_only=True, mode='min',period = 20)
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=10, min_lr = 1e-8) #当标准评估停止提升时，降低学习速率。
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


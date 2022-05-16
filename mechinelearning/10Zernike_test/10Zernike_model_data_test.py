######### parameters setting ########
data_size = [100, 1000, 10000, 17000, 43000, 100000]
model_name = "0511_1"
epoch = [8, 16,20, 500, 400, 300]
batch_size=16
seed = 12333345
data_time = "20220511_104_0.1c"
##data_time = "202103_10_0.1c"
##data_time = "20210305_mix"
input_model = False
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

###################################
class my_callback(Callback):
    def __init__(self,dataGen,showTestDetail=True):
        self.dataGen=dataGen
        self.showTestDetail=showTestDetail
        self.predhis = []
        self.targets = []
    def mape(self,y,predict):
        diff = np.abs(np.array(y) - np.array(predict))
        return np.mean(diff / y)
    def on_epoch_end(self, epoch, logs=None):
        x_test,y_test=next(self.dataGen)
        prediction = self.model.predict(x_test)
        self.predhis.append(prediction)
        #print("Prediction shape: {}".format(prediction.shape))
        #print("Targets shape: {}".format(y_test.shape))
        if self.showTestDetail:
            for index,item in enumerate(prediction):
                print(item,"=====",y_test[index],"====",y_test[index]-item)
        testLoss=self.mape(y_test,prediction)
        print("test loss is :{}".format(testLoss))

############# data and epoch specification #################
intensity_dir = "/home/xianyuer/yuer/num_mechinelearning/tests/32_32/10000/dl_outIntensity.npy"
zernike_dir = "/home/xianyuer/yuer/num_mechinelearning/tests/32_32//10000/dl_zernike_coeff.npy"
#intensity_dir = "/home/lrx/work/10_MO2020/202103+05/intensity_20210305mix.npy"
#zernike_dir = "/home/lrx/work/10_MO2020/202103+05/zernike_20210305mix.npy"
for i in range(0, 3):
    nsnapshot = data_size[i]
    nepoch = epoch[i]
    x = np.load(intensity_dir)
    y = np.load(zernike_dir)
    print("nsnapshot = %s" % nsnapshot)
    print("x shape = ", np.shape(x))
    print("y shape = ", np.shape(y))
    train_x = x[:nsnapshot].reshape([nsnapshot, 32, 32, 1])
    train_y = y[:nsnapshot]

    test_x = x[-1000:].reshape([1000, 32, 32, 1])
    test_y = y[-1000:]
    print("test_x shape = ", np.shape(test_x))
    print("test_y shape = ", np.shape(test_y))
    ############ model specification ##############
    try:
        if(input_model == False):
            model = Xception(input_shape = (32, 32, 1),
                 pooling = 'avg',
                 backend=keras.backend,
                 layers=keras.layers,
                 models=keras.models,
                 utils=keras.utils,
                 middle_loop=0 # middle flux diminished to NONE
        )
        else:
            model = load_model(model_path)
            print("Model loaded from " + model_path)
        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam') #mean_squared_error为损失函数，adam优化器
        model_callback = keras.callbacks.Callback() #回调函数
        modelcheckpoint = ModelCheckpoint(filepath = "./ckpt_models/model.{epoch:02d}.hdf5",
                                        save_best_only=False,
                                        period = 200) #每个训练期之后保存模型。
        reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50, min_lr = 0.0005) #当标准评估停止提升时，降低学习速率。
        batch_print_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch,logs: print(model.predict(train_x)))
        history = History() #把所有事件都记录到 History 对象的回调函数。这个回调函数被自动启用到每一个 Keras 模型。History 对象会被模型的 fit 方法返回。
        callbacks = [modelcheckpoint, reduceLR, history, batch_print_callback]
        
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
            callbacks=callbacks )
    time_end = time.time()
    print("time cost: ", time_end-time_start, "s")

    ############ End model fitting #################

    model.save(model_name+"_"+data_time+"_"+str(nsnapshot)+"_"+str(nepoch)+".h5")
    with open(model_name+"_"+data_time+"_"+str(nsnapshot)+"_"+str(nepoch)+"_history.json", 'w') as f:
        json.dump(history.history, f, cls = NpEncoder)



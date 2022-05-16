######### import session #############
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#import keras.backend as K
#K.set_image_dim_ordering('tf')
import sys
sys.path.append("../")
import time
from keras import losses
from keras import Input
from keras.optimizers import Adam
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
from keras.utils import plot_model
from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
#################################################

model_name = "0919_1_202103_10_0.1c_43000_400.h5"
model = load_model(model_name)
print(model.summary())

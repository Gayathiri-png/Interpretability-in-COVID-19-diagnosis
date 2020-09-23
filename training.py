
import tensorflow as tf
import gc
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
import os, pickle, sys
from timeit import default_timer as timer
from collections import defaultdict
from BBalpha_dropout import *
import pdb
from pdb import set_trace
import pandas as pd

dataset = 'covid'
alpha = 0.5
run = 1

x_train = x_trainn
y_train = y_trainn
x_test = x_testt
y_test = y_testt
y_val = y_valid
x_val= x_valid



# constants
nb_train = x_train.shape[0]
nb_val = x_val.shape[0]
input_dim = (x_train.shape[1], x_train.shape[2])
input_channels = x_train.shape[3]
nb_classes = y_train.shape[1]

batch_size = 2
nb_layers = 2
nb_units = 100
p = 0.5
wd = 1e-6

K_mc = 10

epochs = 30

# model layers
assert K.image_data_format() == 'channels_last', \
        'use a backend with channels last'
input_shape = (input_dim[0], input_dim[1], input_channels) # (dimX, )
inp = Input(shape=input_shape)
layers = get_logit_cnn_layers(nb_units, p, wd, nb_classes, layers = [])

mc_logits = GenerateMCSamples(inp, layers, K_mc) 
loss_function = bbalpha_softmax_cross_entropy_with_mc_logits(alpha)
model = Model(inputs=inp, outputs=mc_logits)
opt = optimizers.Adam(lr=0.00001)
model.compile(optimizer=opt, loss=loss_function,
              metrics=['accuracy', loss_function, metric_avg_acc, metric_avg_ll])

model.summary()
train_Y_dup = np.squeeze(np.concatenate(K_mc * [y_train[:, None]], axis=1)) # N x K_mc x D
val_Y_dup = np.squeeze(np.concatenate(K_mc * [y_val[:, None]], axis=1)) # N x K_mc x D

train_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(x_train)

val_datagen = ImageDataGenerator(rescale=1./255)
val_datagen.fit(x_val)
results = defaultdict(list)
min_val = float('inf')
min_val_ep = 0
ep = 0

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model_bnun.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_datagen.flow(x_train, train_Y_dup,
                        batch_size=batch_size),verbose=1,
                        epochs=10,
                        validation_data=val_datagen.flow(x_val, val_Y_dup),callbacks = [es,mc])

# load the last saved model and add uncertainties in a tf graph
#model.save((os.path.join(directory, 'model.h5')))
directory = '/content/drive/My Drive/covid/data-new-224/new-data-224/saved_models_mod_chk'
filepath = os.path.join(directory, 'best_model_bnun.h5')
K_mc_test = 100
build_test_model(filepath, K_mc_test, p)



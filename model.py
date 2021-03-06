'''
Parts of this code modified from:
Yingzhen Li and Yarin Gal.
Dropout inference in Bayesian neural networks with alpha-divergences.
International Conference on Machine Learning (ICML), 2017.
All credit goes to the original authors
'''

from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Activation, Flatten, Conv2D, MaxPooling2D,Dropout,BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
from keras import metrics
import numpy as np

from keras.models import load_model


# TODO clear comments


###################################################################
# keras aux functions

def Dropout_mc(p, noise_shape=None):
    layer = Lambda(lambda x: K.dropout(x, p, noise_shape=noise_shape),
                   output_shape=lambda shape: shape)
    return layer

def apply_layers(inp, layers):
    output = inp
    for layer in layers:
        output = layer(output) 
    return output

def GenerateMCSamples(inp, layers, K_mc=20, apply_layers=apply_layers):
    if K_mc == 1:
        return apply_layers(inp, layers)
    output_list = []
    for _ in range(K_mc):
        output_list += [apply_layers(inp, layers)]  # THIS IS BAD!!! we create new dense layers at every call!!!!
    def pack_out(output_list):
        #output = K.pack(output_list) # K_mc x nb_batch x nb_classes
        output = K.stack(output_list) # K_mc x nb_batch x nb_classes
        return K.permute_dimensions(output, (1, 0, 2)) # nb_batch x K_mc x nb_classes
    def pack_shape(s):
        s = s[0]
        assert len(s) == 2
        return (s[0], K_mc, s[1])
    # apply pack_out function to output_list, pack_shape to the first item of output_list
    out = Lambda(pack_out, output_shape=pack_shape, name='lambda_pack')(output_list)
    return out

# evaluation for classification tasks
def test_MC_dropout(model, X, Y, from_logits):
    if from_logits:
        mc_logits = model.predict(X)
        mc_log_softmax = mc_logits - np.max(mc_logits, axis=2, keepdims=True)
        mc_log_softmax = mc_log_softmax - np.log(np.sum(np.exp(mc_log_softmax), axis=2, keepdims=True))
    else:
        pred = model.predict(X)
        mc_log_softmax = np.log(pred)
    # mc_log_softmax is now N x K x D
    log_softmax = np.mean(mc_log_softmax, 1) # average over MC samples: N x D
    acc = np.mean(np.argmax(log_softmax, axis=-1) == np.argmax(Y, axis=-1))
    ll = np.mean(np.sum(log_softmax * Y, -1)) # sum over D classes, avg over N samples
    return acc, ll

def logsumexp(x, axis=None):
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def bbalpha_softmax_cross_entropy_with_mc_logits(alpha):
    alpha = K.cast_to_floatx(alpha)
    if alpha != 0.0:
        def bbalpha_loss(y_true, mc_logits):
            # log(p_ij), p_ij = softmax(logit_ij)
            #assert mc_logits.ndim == 3
            mc_log_softmax = mc_logits - K.max(mc_logits, axis=2, keepdims=True)
            mc_log_softmax = mc_log_softmax - K.log(K.sum(K.exp(mc_log_softmax), axis=2, keepdims=True))
            mc_ll = K.sum(y_true * mc_log_softmax, -1)  # N x K
            K_mc = mc_ll.get_shape().as_list()[1]	# only for tensorflow
            # this is the loss function (note inside is also multiplied by alpha
            return - 1. / alpha * (logsumexp(alpha * mc_ll, 1) + K.log(1.0 / K_mc)) 
    else:
        def bbalpha_loss(y_true, mc_logits):
            # this output is N x K, keras will take the mean over N and K
            return K.categorical_crossentropy(y_true, mc_logits, from_logits=True)
    return bbalpha_loss

# custom metrics
def metric_avg_acc(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis =-1)
    avg_pred = K.mean(y_pred, axis=1) # N x D
    y_sample = y_true[:, 0, :] # duplicates : N x D
    acc = K.mean(metrics.categorical_accuracy(y_sample, avg_pred))
    return acc

def metric_avg_ll(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    avg_pred = K.mean(y_pred, axis=1) # N x D
    y_sample = K.mean(y_true, axis=1) # duplicates : N x D
    ll = K.mean(K.log(K.sum(avg_pred * y_sample, axis=-1)))
    return ll  
    

###################################################################
# the model

def get_logit_mlp_layers(nb_layers, nb_units, p, wd, nb_classes, layers = []):
    D = Dropout_mc

    # USING THE LAMBDA FUNCTIONS ENSURES THAT THERE IS DROPOUT AT TEST AND TRAIN TIME

    for _ in range(nb_layers):
        layers.append(D(p))
        layers.append(Dense(nb_units, activation='relu', kernel_regularizer=l2(wd)))
    layers.append(D(p))
    layers.append(Dense(nb_classes, kernel_regularizer=l2(wd))) # these are logit activations!
    return layers

def get_logit_cnn_layers(nb_units, p, wd, nb_classes, layers = []):
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    D = Dropout_mc

    layers.append(Conv2D(64, (kernel_size[0], kernel_size[1]),
                                padding='same', kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Conv2D(64, (kernel_size[0], kernel_size[1]),
                                kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=pool_size))
    
    
    layers.append(Conv2D(128, (kernel_size[0], kernel_size[1]),
                                padding='same', kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Conv2D(128, (kernel_size[0], kernel_size[1]),
                                kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=pool_size))
    
    
    layers.append(Conv2D(256, (kernel_size[0], kernel_size[1]),
                                padding='same', kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Conv2D(256, (kernel_size[0], kernel_size[1]),
                                kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=pool_size))
    
    
    layers.append(Conv2D(512, (kernel_size[0], kernel_size[1]),
                                padding='same', kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Conv2D(512, (kernel_size[0], kernel_size[1]),
                                kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Conv2D(512, (kernel_size[0], kernel_size[1]),
                                kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=pool_size))
   # layers.append(D(0.5))
    
    layers.append(Conv2D(512, (kernel_size[0], kernel_size[1]),
                                padding='same', kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Conv2D(512, (kernel_size[0], kernel_size[1]),
                                kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Conv2D(512, (kernel_size[0], kernel_size[1]),
                                kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=pool_size))
    
    layers.append(Flatten())
    layers.append(Dense(4096, kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(D(0.5))
    
    layers.append(Dense(4096, kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(D(0.5))
     
    layers.append(Dense(nb_classes, kernel_regularizer=l2(wd)))
    return layers

    
###################################################################
# adding build model with K_mc_test MC samples from dropout distribution

def build_test_model(modelpath, K_mc_test, dropout_p):
    model = load_model(modelpath,
                       custom_objects={'bbalpha_loss':
                                       bbalpha_softmax_cross_entropy_with_mc_logits(0.5),
                                       'metric_avg_acc': metric_avg_acc,
                                       'metric_avg_ll': metric_avg_ll})
    
    input_shape = model.layers[0].input_shape[1:] # remove None dimension
    input_shape = (224,224,3)
    inp = Input(shape=input_shape)
    # repeat stochastic layers K_mc_test times (omit input and pack_out layers)
    layers = model.layers[1:-1]
    # noise_shape with 0th dim of 1 keeps same dropout mask across
    # each test sample: the test samples will the same image,
    # with single patches replaced with draws from p(x_i | x_{-i})
    # so same dropout mask needs to be applied for those
    # or it will be a single image, in which case mask is irrelevant
    for ii, layer in enumerate(layers):
        if 'lambda' in layer.name:
            noise_shape = (1,) + layers[ii-1].output_shape[1:] # omit N dim
            layers[ii] = Dropout_mc(dropout_p, noise_shape=noise_shape)
            print("Replacing Layer {}...noise shape: {}"
                  .format(ii, noise_shape))
    mc_logits = GenerateMCSamples(inp, layers, K_mc_test)
    # softmax over the final dim of output
    mc_softmax = Activation('softmax', name='softmax')(mc_logits)
    # output of test_model is N x K_mc_test x C
    test_model = Model(inputs=inp, outputs=mc_softmax)
    output_path = modelpath.rsplit('.', 1)
    output_path = output_path[0] + '-test.' + output_path[-1]
    test_model.save(output_path)

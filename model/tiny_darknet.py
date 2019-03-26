"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2

from model.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def darknet_resblock_body(x, num_filters):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(16, (3,3))(x)
    x = darknet_resblock_body(x, 32)
    x = darknet_resblock_body(x, 64)
    x = darknet_resblock_body(x, 128)
    x = darknet_resblock_body(x, 256)
    x = darknet_resblock_body(x, 512)
    x = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x)
    x = DarknetConv2D_BN_Leaky(1024, (3,3))(x)
    return x

def tiny_resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    for i in range(num_blocks):
        x = DarknetConv2D_BN_Leaky(num_filters//4, (3,3))(x)
        x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    return x

def tiny_darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(16, (3,3))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = tiny_resblock_body(x, 128, 2)
    x = tiny_resblock_body(x, 256, 2)
    x = tiny_resblock_body(x, 512, 2)
    #x = DarknetConv2D_BN_Leaky(128, (3,3))(x)
    return x






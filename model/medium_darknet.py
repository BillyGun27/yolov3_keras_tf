"""YOLO_v3 Model Defined in Keras."""

from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.models import Model
from utils.small_yolo_layer import DarknetConv2D,DarknetConv2D_BN_Leaky,make_last_layers
from utils.utils import compose


def darknet_resblock_body(x, num_filters):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    return x

def darknet_ref_body(x):#tiny yolo with three output
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

def yolo_body(inputs, num_anchors, num_classes):
    #net, endpoint = inception_v2.inception_v2(inputs)
    darknet =  Model(inputs, darknet_ref_body(inputs))

    # input: 416 x 416 x 3
    # leaky_re_lu_7 :13 x 13 x 1024
    # leaky_re_lu_5 :26 x 26 x 512
    # leaky_re_lu_4 : 52 x 52 x 256

    f1 = darknet.get_layer('leaky_re_lu_7').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 256, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f2 = darknet.get_layer('leaky_re_lu_5').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(64, (1,1)),
            UpSampling2D(2))(x)

    f3 = darknet.get_layer('leaky_re_lu_4').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 64, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])



#def tiny_resblock_body(x, num_filters, num_blocks):
#    '''A series of resblocks starting with a downsampling Convolution2D'''
#    # Darknet uses left and top padding instead of 'same' mode
#    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#    for i in range(num_blocks):
#        x = DarknetConv2D_BN_Leaky(num_filters//4, (3,3))(x)
#        x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
#    return x

#def tiny_darknet_body(x):
#    '''Darknent body having 52 Convolution2D layers'''
#    x = DarknetConv2D_BN_Leaky(16, (3,3))(x)
#    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
#    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#    x = tiny_resblock_body(x, 128, 2)
#    x = tiny_resblock_body(x, 256, 2)
#    x = tiny_resblock_body(x, 512, 2)
#    #x = DarknetConv2D_BN_Leaky(128, (3,3))(x)
#    return x






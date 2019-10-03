"""YOLO_v3 Model Defined in Keras."""

from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.models import Model
from utils.yolo_layer import DarknetConv2D,DarknetConv2D_BN_Leaky,make_last_layers
from utils.utils import compose
from keras.applications.densenet import DenseNet121,DenseNet169, DenseNet201

def yolo_body(inputs, num_anchors, num_classes):
    #net, endpoint = inception_v2.inception_v2(inputs)
    densenet = DenseNet121(input_tensor=inputs,weights='imagenet')#include top can be added but will not change much

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 1024
    # conv_pw_5_relu : 52 x 52 x 512

    f1 = densenet.get_layer('relu').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = densenet.get_layer('pool4_relu').output
    # f2: 26 x 26 x 1024 // 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = densenet.get_layer('pool3_relu').output
    # f3 : 52 x 52 x 512 // 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


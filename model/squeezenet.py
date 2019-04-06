
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, GlobalAveragePooling2D, concatenate,Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.models import Model
from utils.small_yolo_layer import DarknetConv2D,DarknetConv2D_BN_Leaky,make_last_layers
from utils.utils import compose
#from keras.utils.data_utils import get_file

#TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/wohlert/keras-squeezenet/raw/master/squeezenet_weights.h5'

# a building block of the SqueezeNet architecture
def fire_module(number, x, squeeze, expand, weight_decay=None, trainable=True):
    
    module_name = 'fire' + number
    
    if trainable and weight_decay is not None:
        kernel_regularizer = l2(weight_decay) 
    else:
        kernel_regularizer = None
    
    x = Convolution2D(
        squeeze, (1, 1), 
        name=module_name + '/' + 'squeeze',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    x = Activation('relu')(x)

    a = Convolution2D(
        expand, (1, 1),
        name=module_name + '/' + 'expand1x1',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    a = Activation('relu')(a)

    b = Convolution2D(
        expand, (3, 3), padding='same',
        name=module_name + '/' + 'expand3x3',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    b = Activation('relu')(b)

    return concatenate([a, b])


def squeezenet_body(weight_decay=1e-4, input_tensor=Input(shape=(416, 416, 3))):

    image = input_tensor

    x = ZeroPadding2D(((1,0),(1,0)))(image)

    x = Convolution2D(
        64, (3, 3), strides=(2, 2), name='conv1', 
        trainable=True
    )(x) # 111, 111, 64
    
    x = Activation('relu')(x)
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 207,207,64

    x = fire_module('2', x, squeeze=16, expand=64) # 207,207,64
    x = fire_module('3', x, squeeze=16, expand=64) # 207,207,64
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 27, 27, 128

    x = fire_module('4', x, squeeze=32, expand=128) # 103,103,128
    x = fire_module('5', x, squeeze=32, expand=128) # 103,103,128
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 51, 51, 128
    
    x = fire_module('6', x, squeeze=48, expand=192) # 25, 25, 384
    x = fire_module('7', x, squeeze=48, expand=192) # 25, 25, 384
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 12, 12, 384

    x = fire_module('8', x, squeeze=64, expand=256) # 12, 12, 512
    x = fire_module('9', x, squeeze=64, expand=256) # 12, 12, 512
    x = fire_module('10', x, squeeze=128, expand=320) # 12, 12, 640
    x = fire_module('11', x, squeeze=128, expand=320) # 12, 12, 640
    
    '''
    x = Dropout(0.5)(x)
    x = Convolution2D(
        320, (1, 1), name='conv10',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(x) # 13, 13, 256
    '''
    #x = Activation('relu')(x)
    #logits = GlobalAveragePooling2D()(x) # 256
    #probabilities = Activation('softmax')(logits)
    
    model = Model(image, x )
       # load weights
    #weights_path = get_file(
    #    'squeezenet_weights.h5',
    #    TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models'
    #)
    #model.load_weights(weights_path, skip_mismatch=True)
    
    return model

def yolo_body(inputs, num_anchors, num_classes):
    #net, endpoint = inception_v2.inception_v2(inputs)
    squeezenet = squeezenet_body(input_tensor=inputs)

    # input: 416 x 416 x 3
    # contatenate_10 :12 x 12 x 640
    # contatenate_6 :25 x 25 x 384
    # contatenate_4 : 51 x 51 x 256

    f1 = squeezenet.get_layer('concatenate_10').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = squeezenet.get_layer('concatenate_6').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = squeezenet.get_layer('concatenate_4').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])
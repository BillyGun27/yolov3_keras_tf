import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import Input, Lambda ,Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.core import preprocess_true_boxes, yolo_loss
from model.mobilenet import mobilenet_yolo_body,mobilenetv2_yolo_body
from model.mobilenet import mobilenetv2_yolo_body
from model.small_mobilenet import mobilenetv2_yolo_body
from model.utils  import get_random_data
from keras.utils.vis_utils import plot_model as plot
from model.squeezenet import squeezenet_body,squeezenet_yolo_body
from model.tiny_darknet import tiny_darknet_body,darknet_body
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
    
train_path = '2007_train.txt'
val_path = '2007_val.txt'
# test_path = '2007_test.txt'
log_dir = 'logs/logits_only_000/'
classes_path = 'class/voc_classes.txt'
anchors_path = 'anchors/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416) # multiple of 32, hw
    
  
image_input = Input(shape=(416, 416, 3))
h, w = input_shape
num_anchors = len(anchors)

#mobilenet_model = MobileNet(input_tensor=image_input,weights='imagenet')
#mobilenet_model = mobilenet_yolo_body(image_input, num_anchors//3, num_classes)
#plot(model, to_file='{}.png'.format("mobilenet_yolo"), show_shapes=True)
#mobilenet_model.summary()
#mobilenet_model.save_weights('empty_mobilenet.h5')

<<<<<<< HEAD
#mobilenetv2_model = MobileNetV2(input_tensor=image_input,weights='imagenet')
#mobilenetv2_model = mobilenetv2_yolo_body(image_input, num_anchors//3, num_classes)
#plot(model, to_file='{}.png'.format("mobilenet_yolo"), show_shapes=True)
#mobilenetv2_model.summary()
#mobilenetv2_model.save_weights('empty_mobilenetv2.h5')

#squeezenet_model = squeezenet_body( input_tensor = image_input )
=======
mobilenetv2 = mobilenetv2_yolo_body(image_input, num_anchors//3, num_classes)
mobilenetv2.summary()
mobilenetv2.save_weights('empty_mobilenet.h5')
plot(mobilenetv2, to_file='{}.png'.format("mobilenetv2_yolo"), show_shapes=True)

#squeezenet_model = squeezenet_body( input_tensor = image_input )
#squeezenet_model.summary()
>>>>>>> 7762d922d53f77fb0da639b9f9434d15627e3608
#squeezenet_model = squeezenet_yolo_body(image_input, num_anchors//3, num_classes)
#plot(squeezenet_model , to_file='{}.png'.format("squeezenet_yolo"), show_shapes=True)
#squeezenet_model.summary()
#squeezenet_model.save_weights('empty_squeezenet.h5')


#tiny_model = tiny_darknet_yolo_body(image_input, num_anchors//3, num_classes)
#tiny_model.summary()

#darknet = Model( image_input , darknet_body(image_input ))
#plot(darknet , to_file='{}.png'.format("darknet_body"), show_shapes=True)
#darknet.summary()
#darknet.save_weights('empty_darknet.h5')

#tiny_darknet = Model( image_input , tiny_darknet_body(image_input ))
#plot(tiny_darknet , to_file='{}.png'.format("tiny_darknet_body"), show_shapes=True)
#tiny_darknet.summary()
#tiny_darknet.save_weights('empty_tiny_darknet.h5')


#### 2 scale only
#mobilenet_model = MobileNet(input_tensor=image_input,weights='imagenet')
mobilenet_model = mobilenet_yolo_body(image_input, num_anchors//3, num_classes)
yolo3 = Reshape((13, 13, 3, 25))(mobilenet_model.layers[-3].output)
yolo2 = Reshape((26, 26, 3, 25))(mobilenet_model.layers[-2].output)
yolo1 = Reshape((52, 52, 3, 25))(mobilenet_model.layers[-1].output)

mobilenet_model = Model( inputs= mobilenet_model.input , outputs=[yolo1] )
#plot(mobilenet_model, to_file='{}.png'.format("normal_mobilenet_yolo"), show_shapes=True)
mobilenet_model.summary()
mobilenet_model.save_weights('empty_little_mobilenet.h5')



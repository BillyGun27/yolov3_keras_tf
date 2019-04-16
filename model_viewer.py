import numpy as np
from utils.core import preprocess_true_boxes, yolo_loss
from keras.models import Model,load_model
from keras.layers import Input
from keras.utils.vis_utils import plot_model as plot

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2

#from model.new_squeezenet import SqueezeNet,yolo_body
#from model.small_mobilenets2 import yolo_body
#from model.mobilenet import yolo_body
from model.mobilenetv2 import yolo_body
#from model.medium_darknet import darknet_ref_body,yolo_body
#from model.yolo3 import darknet_body, yolo_body, tiny_yolo_body


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
    
  
image_input = Input(shape=(416,416, 3))
h, w = input_shape
num_anchors = len(anchors)

#darknet = Model( image_input ,  darknet_ref_body(image_input ) )
#darknet = yolo_body(image_input, num_anchors//3, num_classes)
#plot(darknet , to_file='{}.png'.format("darknet_ref_yolo2"), show_shapes=True)
#darknet.summary()
#print(len(darknet.layers))
#darknet.save_weights('empty_medium_yolo2.h5')

#squeezenet_model = squeezenet_body(weight_decay=1e-4, input_tensor=Input(shape=(416, 416, 3)))
#squeezenet_model = SqueezeNet(input_tensor=image_input,weights='imagenet',include_top=False)
#squeezenet_model = yolo_body(image_input, num_anchors//3, num_classes)
#plot(squeezenet_model, to_file='{}.png'.format("squeezenet_yolo"), show_shapes=True)
#squeezenet_model.summary()
#print(len(squeezenet_model.layers))
#squeezenet_model.save_weights('empty_squeezenet.h5')

#mobilenet_model = MobileNetV2(input_tensor=image_input,weights='imagenet' ,include_top=False)
#mobilenet_model = yolo_body(image_input, num_anchors//3, num_classes)
#plot(mobilenet_model, to_file='{}.png'.format("new_small_mobilenets2_yolo"), show_shapes=True)
#mobilenet_model.summary()
#print(len(mobilenet_model.output))
#print(len(mobilenet_model.layers))
#mobilenet_model.save_weights('empty_mobilenet.h5')


#mobilenetv2_model = MobileNetV2(input_tensor=image_input,weights='imagenet',include_top=False)
#mobilenetv2_model = mobilenetv2_yolo_body(image_input, num_anchors//3, num_classes)
#plot(mobilenetv2_model, to_file='{}.png'.format("mobilenet_yolov2"), show_shapes=True)
#mobilenetv2_model.summary()
#print(len(mobilenetv2_model.layers))
#mobilenetv2_model.save_weights('empty_mobilenetv2.h5')

#darknet = Model( image_input ,  darknet_body(image_input) )
#darknet = yolo_body(image_input, num_anchors//3, num_classes)
#plot(darknet , to_file='{}.png'.format("darknet53_yolo"), show_shapes=True)
#darknet.summary()
#darknet.save_weights('empty_darknet_body.h5')

#darknet = tiny_yolo_body(image_input, num_anchors//3, num_classes)
#plot(darknet , to_file='{}.png'.format("tiny_yolo"), show_shapes=True)
#darknet.summary()
#print(len(darknet.layers))
#darknet.save_weights('empty_tiny_yolo.h5')

model = load_model("model_data/416bnfuse_small_mobilenets2_trained_model.h5") 
model.summary()
print(len(model.layers))




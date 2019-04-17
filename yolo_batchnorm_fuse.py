import numpy as np
from utils.core import preprocess_true_boxes, yolo_loss
from keras.models import Model
from keras.layers import Input
from keras.utils.vis_utils import plot_model as plot
from kito import reduce_keras_model

#from model.small_mobilenets2 import yolo_body
from model.yolo3 import tiny_yolo_body as yolo_body

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

#model_path = 'model_data/new_small_mobilenets2_trained_weights_final.h5'
model_path = 'model_data/tiny_yolo.h5'

train_path = '2007_train.txt'
val_path = '2007_val.txt'
# test_path = '2007_test.txt'
log_dir = 'logs/logits_only_000/'
classes_path = 'class/coco_classes.txt'
anchors_path = 'anchors/tiny_yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

#input_shape = (416,416) # multiple of 32, hw
    
  
image_input = Input(shape=(416,416, 3))
#h, w = input_shape
num_anchors = len(anchors)

model = yolo_body(image_input, 3, num_classes)
model.load_weights(model_path)
model.summary()
print(len(model.layers))

model_reduced = reduce_keras_model(model)
#model_reduced.save('model_data/416bnfuse_small_mobilenets2_trained_model.h5')
#model_reduced.save('model_data/416bnfuse_tiny_yolo.h5')
model_reduced.summary()
print(len(model_reduced.layers))



from keras.models import Model
from keras.layers import Input
from utils.train_tool import get_classes,get_anchors
from model.small_mobilenet import yolo_body

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model_path = 'model_data/small_mobilenet_trained_weights_final.h5'
classes_path = 'class/voc_classes.txt'
anchors_path = 'anchors/yolo_anchors.txt'

class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)

num_classes = len(class_names)
num_anchors = len(anchors)
        
yolo_model =  yolo_body(Input(shape=(416,416,3)), num_anchors//3, num_classes)
yolo_model.load_weights(model_path)

yolo_model.save('model_data/small_mobilenet_trained_model.h5')

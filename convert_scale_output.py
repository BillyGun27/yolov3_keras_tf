from keras.models import Model,load_model
from keras.layers import Input
from utils.train_tool import get_classes,get_anchors
from model.small_mobilenet import yolo_body
from model.yolo3 import tiny_yolo_body

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#model_path = 'model_data/small_mobilenet_trained_weights_final.h5'
model_path = 'model_data/tiny_yolo.h5'
classes_path = 'class/coco_classes.txt'
anchors_path = 'anchors/tiny_yolo_anchors.txt'

class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)

num_classes = len(class_names)
num_anchors = len(anchors)
        
yolo_model =  tiny_yolo_body(Input(shape=(416,416,3)), 3, num_classes)
yolo_model.load_weights(model_path)
#yolo_model = load_model(model_path)

yolo_model.summary()

yolo3 = yolo_model.layers[-3].output
yolo2 = yolo_model.layers[-2].output
#yolo1 = yolo_model.layers[-1].output

#new_model = Model( inputs= yolo_model.input , outputs=[yolo3,yolo2] )
#new_model = Model( inputs= yolo_model.input , outputs=[yolo3] )
new_model = Model( inputs= yolo_model.input , outputs=[yolo2] )#for tiny yolo
new_model.summary()
#new_model.save('model_data/2scale_small_mobilenet_trained_model.h5')
#new_model.save('model_data/1scale_small_mobilenet_trained_model.h5')
#new_model.save('model_data/1scale_tiny_yolo_model.h5') 


#model_path = 'model_data/tiny_yolo.h5'
#model_path = 'model_data/1scale_tiny_yolo_model.h5'
#yolo_model = load_model(model_path)
#yolo_model.summary()
#print(len(yolo_model.output))


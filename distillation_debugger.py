"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 
from keras.layers import Input, Lambda ,Reshape
from utils.core import yolo_head,box_iou

from utils.core import preprocess_true_boxes
from utils.utils  import get_random_data
from utils.train_tool import get_classes,get_anchors
from utils.evaluation import AveragePrecision

from utils.train_tool import get_classes,get_anchors

#changeable param
from model.small_mobilenets2 import yolo_body
from model.yolo3 import yolo_body as teacher_body, tiny_yolo_body

import argparse

def _main():
    epoch_end_first = 20
    epoch_end_final = 2
    model_name = 'test_loss_basic_distill_mobilenet'
    log_dir = 'logs/test_loss_basic_distill_mobilenet_000/'
    model_path = 'model_data/new_small_mobilenets2_trained_weights_final.h5'
    #teacher_path = 'model_data/new_yolo_trained_weights_final.h5'
    teacher_path = 'model_data/trained_weights_final.h5'

    train_path = '2007_train.txt'
    val_path = '2007_val.txt'
   # test_path = '2007_test.txt'
    classes_path = 'class/voc_classes.txt'
    anchors_path = 'anchors/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw
    num_anchors = len(anchors)
    image_input = Input(shape=(None, None, 3))

    with open(train_path) as f:
        train_lines = f.readlines()
    train_lines = train_lines[:4]

    with open(val_path) as f:
        val_lines = f.readlines()
    val_lines = val_lines[:4]

   # with open(test_path) as f:
   #     test_lines = f.readlines()

    num_train = int(len(train_lines))
    num_val = int(len(val_lines))


    #declare model
    num_anchors = len(anchors)
    image_input = Input(shape=(416, 416, 3))
    teacher = teacher_body(image_input, num_anchors//3, num_classes)
    teacher.load_weights(teacher_path)


    batch_size = 1
    datagen =  data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes,teacher)
    logits , zero = next(datagen)

    print(logits[1].shape)
    #print(logits[1])
    arrp = logits[1]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print(logits[1][tuple(box[0])])
       

    print(logits[4].shape)
    arrp = logits[4]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print(logits[4][tuple(box[0])])


def sigmoid(x):
        """sigmoid.

        # Arguments
            x: Tensor.

        # Returns
            numpy ndarray.
        """
        return 1 / (1 + np.exp(-x))


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    '''data generator for fit_generator'''

    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []

        for b in range(batch_size):
            #if i==0:
            #    np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        m_true = teacher.predict(image_data)
        
        #print( teacher.layers[-6].get_weights()[0][0][0][0][0] )
        #print ( teacher.layers[-6].get_weights()[1][0] )

        h, w = input_shape
        num_anchors = len(anchors)
        
        l_true =  [ np.zeros( shape=( batch_size ,416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], 9//3, 20+5) ) for l in range(3) ]

        #print(len(y_true))
        #print(len(m_true))
        #print(len(l_true))
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 

        

        for l in range( len(m_true) ) : 
            object_mask = tf.Variable( y_true[l][..., 4:5] )

            pred_output = tf.Variable(m_true[l]) 
            anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 
            pred_xy, pred_wh , pred_conf , pred_class = yolo_head( pred_output ,anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=False)
            pred_model = K.concatenate([pred_xy, pred_wh, pred_conf ,pred_class  ])
            
            pred_model =  K.switch(object_mask, pred_model , K.zeros_like(pred_model))
            

            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                print( K.shape(pred_output).eval() )
                print( K.shape(object_mask).eval() )
                l_true[l] = pred_model.eval()

            '''
            anchors_tensor = np.reshape( anchors[anchor_mask[l]] , [1, 1, 1, len( anchors[anchor_mask[l]] ) , 2] )

            grid_shape = m_true[l].shape[1:3] # height, width
            grid_shape
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                [grid_shape[0], 1, 1, 1])
            grid = np.concatenate([grid_x, grid_y],axis=-1)

            #print(l)
            m_true[l][...,:2] = (sigmoid(m_true[l][...,:2]) + grid ) / np.array( grid_shape[::-1] )
            m_true[l][..., 2:4] = np.exp(m_true[l][..., 2:4]) * anchors_tensor  / np.array( input_shape[::-1] )
            m_true[l][..., 4] = sigmoid(m_true[l][..., 4])
            m_true[l][..., 5:] = sigmoid(m_true[l][..., 5:])
            
            #print("inside")
            box = np.where(y_true[l][...,4] > 0.5 )
            box = np.transpose(box)

            for i in range(len(box)):
                #if m_true[l][..., 4] > 0.5 :
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
            '''
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)


if __name__ == '__main__':
    _main()


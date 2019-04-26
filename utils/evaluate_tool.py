"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
from keras.layers import Input,Reshape
from keras.models import Model,load_model

from utils.core import preprocess_true_boxes, yolo_loss
from utils.utils  import get_random_data
from tqdm import tqdm

from model.yolo3 import tiny_yolo_body

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
        #print("data"+str(i) )
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
        #l_true =  [ np.zeros( shape=( batch_size ,416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], 9//3, 20+5) ) for l in range(3) ]

        if len(m_true)==3 :
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] 
        elif len(m_true)==2 :
            anchor_mask =  [[3,4,5], [0,1,2]]
        else :
            anchor_mask = [[0,1,2]]

        for l in range( len(m_true) ) : 

            pred_xy, pred_wh , pred_conf , pred_class = numpy_yolo_head( m_true[l] ,anchors[anchor_mask[l]], input_shape )
            pred_detect = np.concatenate([pred_xy, pred_wh , pred_conf , pred_class ],axis=-1)

            #print("inside")
            box = np.where(y_true[l][...,4] > 0.5 )
            box = np.transpose(box)
            for k in range(len(box)):
                m_true[l][tuple(box[k])] = pred_detect[tuple(box[k])] 

        yield image_data, y_true ,m_true

def data_generator_wrapper_eval(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)

def numpy_yolo_head( pred_raw , anchors_m , input_shape ):                                                    
    anchors_tensor = np.reshape( anchors_m , [1, 1, 1, len( anchors_m ) , 2] )

    grid_shape = pred_raw.shape[1:3] # height, width
    grid_shape
    grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y],axis=-1)

    #print(l)
    pred_xy = (sigmoid(pred_raw [...,:2]) + grid ) / np.array( grid_shape[::-1] )
    pred_wh = np.exp(pred_raw [..., 2:4]) * anchors_tensor  / np.array( input_shape[::-1] )
    pred_conf = sigmoid(pred_raw[..., 4:5])
    pred_class = sigmoid(pred_raw[..., 5:])

    return pred_xy, pred_wh , pred_conf , pred_class

def numpy_box_iou(b1, b2):
    # Expand dim to apply broadcasting.
    b1 = np.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = np.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap  

    '''
    for i in range(len(box)):
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print( "({})".format(box[i]) )
    print( flogits[0][tuple(box[i])][0:5] )
    print( flogits[0][tuple(box[i])][5:25] )
    true_label =  np.argmax( flogits[0][tuple(box[i])][5:25]) 
    print( "{} = {}".format(true_label, class_names[ true_label ] ) )
    print("-------------------------------------------------------")
    print( mlogits[0][ tuple(box[i]) ][0:5] )
    print( mlogits[0][ tuple(box[i]) ][5:25] )
    pred_label =  np.argmax( flogits[0][tuple(box[i])][5:25]) 
    print( "{} = {}".format(pred_label, class_names[ pred_label ] ) )
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    '''
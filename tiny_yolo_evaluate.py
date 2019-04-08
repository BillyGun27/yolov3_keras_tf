"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
from keras.layers import Input,Reshape
from keras.models import Model

from utils.core import preprocess_true_boxes, yolo_loss
from utils.utils  import get_random_data
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from model.yolo3 import tiny_yolo_body

#from model.yolo3 import yolo_body
#from model.mobilenet import yolo_body
#from model.small_mobilenet import yolo_body
#from model.squeezenet import yolo_body


import argparse

def _main():
    #weights_path = 'model_data/trained_weights_final_mobilenet.h5'
    #weights_path = 'model_data/trained_weights_final_small_mobilenet.h5'
   # weights_path = 'logs/tiny_yolo_000/ep027-loss28.354-val_loss27.772.h5'
    weights_path = 'logs/tiny_yolo_000/tiny_yolo_trained_weights_stage_1.h5'
    #weights_path = 'logs/tiny_yolo_000/ep060-loss25.849-val_loss26.280.h5'

    
    train_path = '2007_train.txt'
    val_path = '2007_val.txt'
    test_path = '2007_test.txt'
    #log_dir = 'logs/logits_only_000/'
    classes_path = 'class/voc_classes.txt'
    anchors_path = 'anchors/tiny_yolo_anchors.txt'
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    
    input_shape = (416,416) # multiple of 32, hw
    

    with open(train_path) as f:
        train_lines = f.readlines()

    with open(val_path) as f:
        val_lines = f.readlines()

    with open(test_path) as f:
        test_lines = f.readlines()


    num_val = int(len(train_lines))
    num_train = int(len(val_lines))
    num_test = int(len(test_lines))

    #declare model
    num_anchors = len(anchors)
    #num_anchors = 6
    #num_anchors//2 == 3
    image_input = Input(shape=(416, 416, 3))
    eval_model = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    eval_model.load_weights(weights_path)
    
    # return the constructed network architecture
    # class+5
    yolo2 = Reshape((13, 13, 3, 25))(eval_model.layers[-2].output)
    yolo1 = Reshape((26, 26, 3, 25))(eval_model.layers[-1].output)
 #   yolo1 = Reshape((52, 52, 3, 25))(eval_model.layers[-1].output)
    
    eval_model = Model( inputs= eval_model.input , outputs=[yolo2,yolo1] )
    eval_model._make_predict_function()

    batch_size = 1
   
    all_detections  = [ [] for i in range(num_classes) ]
    all_annotations = [ [] for i in range(num_classes) ]
    
    num_layers = num_anchors//3
    #print(num_layers)
    
    count_detections  = [ [0 for i in range(num_classes)] for i in range(num_layers) ]
    
    
    datagen = data_generator_wrapper(test_lines, batch_size, input_shape, anchors, num_classes,eval_model)
    
    
    print( "{} test data".format(num_test) )
    for n in tqdm( range(num_test) ):#num_test
        img,flogits,mlogits = next(datagen)

        for l in range(num_layers):
            #print( "layer" + str(l) )
            arrp = flogits[l]
            box = np.where(arrp[...,4] > 0 )
            box = np.transpose(box)

            for i in range(len(box)):
                #print("obj" + str(i) )
                #detection_label =  np.argmax( flogits[l][tuple(box[i])][5:]) 
                annotation_label =  np.argmax( flogits[l][tuple(box[i])][5:]) 

                #print( "{} ({}) {} == ({}) {} ".format(l, detection_label, class_names[  detection_label ] ,annotation_label, class_names[  annotation_label ] ) )
                
                all_detections[annotation_label].append( mlogits[l][tuple(box[i])] ) 
                all_annotations[annotation_label].append( flogits[l][tuple(box[i])] )

                count_detections[l][annotation_label] +=1
    

    print(len(all_detections) )
    print(len(all_annotations) )
    print(count_detections)

    iou_thres = 0.5
    conf_thres = 0.5
    
    average_precisions = {}

    for label in tqdm( range( num_classes ) ) : 
        
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))

        
        num_detect = len( all_detections[label] )
        for det in  range( num_detect ):

            detect_box = all_detections[label][det][...,0:4]
            detect_conf = all_detections[label][det][...,4]
            detect_label =  np.argmax( all_detections[label][det][...,5:] ) 

            annot_box = all_annotations[label][det][...,0:4]
            annot_conf = all_annotations[label][det][...,4]
            detect_label =  np.argmax( all_detections[label][det][...,5:] ) 
            
            iou = numpy_box_iou( detect_box , annot_box)

            scores = np.append(scores, detect_conf )

        
            if( iou > iou_thres and  detect_conf > conf_thres and (label == detect_label ) ):
                #print( best_iou[tuple(box[i])] )
                #print("pos")
                false_positives = np.append(false_positives, 0)
                true_positives   = np.append(true_positives, 1)
            else:
                #print("neg")
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)
                
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]
        #print(true_positives)

        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)
        #print(true_positives)

        recall = true_positives  / num_detect
        #print( recall )
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        #print( precision )

        average_precision  = compute_ap(recall, precision)
        average_precisions[label] = average_precision
    
    print( "loaded weights {}".format(weights_path) )

    #print(average_precisions)

    for label, average_precision in average_precisions.items():
        print(class_names[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           
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

        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 

        for l in range( len(m_true) ) : 

            pred_xy, pred_wh , pred_conf , pred_class = numpy_yolo_head( m_true[l] ,anchors[anchor_mask[l]], input_shape )
            pred_detect = np.concatenate([pred_xy, pred_wh , pred_conf , pred_class ],axis=-1)

            #print("inside")
            box = np.where(y_true[l][...,4] > 0.5 )
            box = np.transpose(box)
            for k in range(len(box)):
                m_true[l][tuple(box[k])] = pred_detect[tuple(box[k])] 

        yield image_data, y_true ,m_true

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
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

if __name__ == '__main__':
    _main()

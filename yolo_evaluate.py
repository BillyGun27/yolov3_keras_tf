"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
from keras.layers import Input,Reshape
from keras.models import Model,load_model

from utils.core import preprocess_true_boxes, yolo_loss
from utils.utils  import get_random_data
from utils.train_tool import get_classes,get_anchors
from utils.evaluate_tool import sigmoid, data_generator_wrapper_eval,numpy_box_iou,compute_ap
from tqdm import tqdm

from model.yolo3 import tiny_yolo_body

#from model.yolo3 import yolo_body
#from model.mobilenet import yolo_body
from model.small_mobilenets2 import yolo_body
#from model.squeezenet import yolo_body


import argparse

def _main():
    #weights_path = 'model_data/trained_weights_final_mobilenet.h5'
    weights_path = 'model_data/small_mobilenets2_trained_weights_final.h5'
    #weights_path = 'logs/squeezenet_000/squeezenet_trained_weights_final.h5'
    
    train_path = '2007_train.txt'
    val_path = '2007_val.txt'
    test_path = '2007_test.txt'
    #log_dir = 'logs/logits_only_000/'
    classes_path = 'class/voc_classes.txt'
    anchors_path = 'anchors/yolo_anchors.txt'
    
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    num_classes = len(class_names)
    num_anchors = len(anchors) #9

    shape_size = 416
    input_shape = (shape_size, shape_size) # multiple of 32, hw

    num_layers = num_anchors//3 #9//3

    #with open(train_path) as f:
    #    train_lines = f.readlines()
    #train_lines = train_lines[:200]

    #with open(val_path) as f:
    #    val_lines = f.readlines()
    #val_lines = val_lines[:150]

    with open(test_path) as f:
        test_lines = f.readlines()
    #test_lines = test_lines[:2]


    #num_train = int(len(train_lines))
    #num_val = int(len(val_lines))
    num_test = int(len(test_lines))

    #declare model
   
    image_input = Input(shape=(shape_size, shape_size, 3))
    
    try:
            eval_model = load_model(model_path, compile=False)
    except:
            eval_model = yolo_body(image_input, num_anchors//3, num_classes)#9//3
            eval_model.load_weights(weights_path)

    
    yolo_out = []
    fmap = shape_size//32
    mapsize = [1,2,4]

    if num_layers==3 :
        ly_out = [-3, -2, -1] 
    elif num_layers==2 :
        ly_out = [-2, -1] 
    else :
        ly_out = [-1] 

    # return the constructed network architecture
    # class+5
    for ly in range(num_layers):
        yolo_layer = Reshape(( fmap*mapsize[ly], fmap*mapsize[ly] , 3, (num_classes + 5) ))(eval_model.layers[ ly_out[ly] ].output)

        yolo_out.append(yolo_layer)
    
    eval_model = Model( inputs= eval_model.input , outputs= yolo_out )
    eval_model._make_predict_function()

    batch_size = 1
   
    all_detections  = [ [] for i in range(num_classes) ]
    all_annotations = [ [] for i in range(num_classes) ]

    count_detections  = [ [0 for i in range(num_classes)] for i in range(num_layers) ]
    total_object = 0

    datagen = data_generator_wrapper_eval(test_lines, batch_size, input_shape, anchors, num_classes,eval_model)
    
    
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
                #print( tuple(box[i]) )
                #detection_label =  np.argmax( flogits[l][tuple(box[i])][5:]) 
                annotation_label =  np.argmax( flogits[l][tuple(box[i])][5:]) 

                #print( "{} ({}) {} == ({}) {} ".format(l, detection_label, class_names[  detection_label ] ,annotation_label, class_names[  annotation_label ] ) )
                
                all_detections[annotation_label].append( mlogits[l][tuple(box[i])] ) 
                all_annotations[annotation_label].append( flogits[l][tuple(box[i])] )

                count_detections[l][annotation_label] +=1
                total_object +=1
    

    print(len(all_detections) )
    print(len(all_annotations) )
    print(count_detections)
    print(total_object)

    
    conf_thres = 0.5
    iou_thres = 0.45
    
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
    
  
#0661
if __name__ == '__main__':
    _main()

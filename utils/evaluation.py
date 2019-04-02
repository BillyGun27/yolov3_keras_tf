import numpy as np
import tensorflow as tf
from keras.layers import  Reshape
from keras.models import Model
from keras.callbacks import Callback
from tqdm import tqdm

class AveragePrecision(Callback):
        def __init__(self, val_data ,total_image,input_shape,num_layers,anchors,num_classes,log_dir):
            super().__init__()
            self.validation_data = val_data
            self.total_image = total_image
            self.input_shape = input_shape
            self.num_layers = num_layers
            self.anchors = anchors
            self.num_classes = num_classes
            self.writer = tf.summary.FileWriter(log_dir)

        def on_epoch_begin(self, epoch, logs={}):
            self.losses = []
            #print(self.validation_data)
            #print( self.caller("b") )
            #print( self.batch_size )
        
        #this code only evaluate label that appear in the image not all label
        def on_epoch_end(self, epoch, logs={}):
            #self.losses.append(logs.get('loss'))
          
            #obj = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
            all_map = []
            for b in tqdm( range(self.total_image) ):
                layers_map = []
                #print("batch" + str(b) )
                val_dat , zeros = next( self.validation_data )
                image_data = val_dat[0] 
                true_label = val_dat[1:4] if self.num_layers==3 else val_dat[1:3]

                #print( true_label[0].shape )
                #print( true_label[1].shape )
                #print( true_label[2].shape )

                anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if self.num_layers==3 else [[3,4,5], [1,2,3]]
                
                scale_map = []
                for lyr in range(self.num_layers):
                    #print(self.num_layers)
                    #print("layer" + str(lyr) )
                    #print( true_label[lyr].shape )
                    arrp = true_label[lyr]
                    box = np.where(arrp[...,4] > 0 )
                    box = np.transpose(box)
                    
                    #print(box)
                    #print("box" + str( len(box) ) ) 
                    if( len(box) > 0 ):
                        
                        s = true_label[lyr].shape[1]
                        layer_name = "conv2d_output_" + str ( s )
                        num_anchors = 3
                        endlayer = Reshape( (s, s, num_anchors, self.num_classes+5 ) )( self.model.get_layer(layer_name).output )
                        testmodel =  Model(  self.model.layers[0].input ,  endlayer  )
                        pred_output= testmodel.predict( image_data )
                        #print(pred_output.shape)
                        

                        pred_xy, pred_wh , pred_conf , pred_class = numpy_yolo_head( pred_output ,self.anchors[anchor_mask[lyr]], self.input_shape )
                        pred_box = np.concatenate([pred_xy, pred_wh],axis=-1)
                        
                        #print(pred_box.shape)

                        #### Measure AP #########################################

                        #measure iou
                        object_mask = arrp[..., 4:5]
                        object_mask_bool = np.array(object_mask , dtype=bool)
                        true_box = arrp[0,...,0:4][ object_mask_bool[0,...,0] ]
                        #print(true_box.shape)
                        iou = numpy_box_iou(pred_box, true_box)
                        best_iou = np.max(iou, axis=-1)
                        

                        iou_thres = 0.5
                        conf_thres = 0.5
                        false_positives = np.zeros((0,))
                        true_positives  = np.zeros((0,))
                        scores          = np.zeros((0,))

                        for i in range(len(box)):
                            true_class_label =  np.argmax( arrp[tuple(box[i])][5:25]) 
                            pred_class_label =  np.argmax( pred_class[tuple(box[i])]) 
                            scores = np.append(scores, pred_conf[ tuple(box[i]) ] )
                            if( best_iou[tuple(box[i])] > iou_thres and  pred_conf[tuple(box[i])] > conf_thres and (true_class_label == pred_class_label ) ):
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

                        recall = true_positives  / len(box)
                        #print( recall )
                        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
                        #print( precision )

                        average_precision  = compute_ap(recall, precision)

                        ###########################################################################

                        #print(average_precision)
                        scale_map.append(average_precision)
                        
                
                #print(np.mean(scale_map))
                if( scale_map ):
                    all_map.append( np.mean(scale_map) )
                else :
                     all_map.append( 0 )
                

            #print("batch")
            if( all_map ):
                    mAPvalue = np.mean(all_map)
            else :
                    print("no object")
                    mAPvalue = 0
            
            print("mAP : " + str( mAPvalue ) )
            summary = tf.Summary(value=[tf.Summary.Value(tag='mAP', simple_value=mAPvalue)])
            self.writer.add_summary(summary, epoch)

def sigmoid(x):
        """sigmoid.

        # Arguments
            x: Tensor.

        # Returns
            numpy ndarray.
        """
        return 1 / (1 + np.exp(-x))

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
    pred_conf = sigmoid(pred_raw[..., 4])
    pred_class = sigmoid(pred_raw[..., 5:])

    return pred_xy, pred_wh , pred_conf , pred_class

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
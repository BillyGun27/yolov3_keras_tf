import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from utils.utils  import get_random_data
from utils.core import yolo_head,box_iou
from utils.core import preprocess_true_boxes


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
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        m_true = teacher.predict(image_data)
         
        h, w = input_shape
        num_anchors = len(anchors)
        
        l_true =  [ np.zeros( shape=( batch_size ,416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], 9//3, 20+5) ) for l in range(3) ]

        #print(len(y_true))
        #print(len(m_true))
        #print(len(l_true))
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 

        for l in range( len(m_true) ) : 
            '''
            pred_output = tf.Variable(m_true[l]) 
            anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 
            pred_xy, pred_wh , pred_conf , pred_class = yolo_head( pred_output ,anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=False)
            pred_model = K.concatenate([pred_xy, pred_wh, pred_conf ,pred_class  ])

            with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    
                    pred_model = pred_model.eval()
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
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
        
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)

def distill_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)


def darknet_raw(feats,object_mask,grid_shape,grid,input_shape,anchors,anchor_msk):
    raw_true_xy = feats[..., :2]*grid_shape[::-1] - grid
    raw_true_wh = K.log(feats[..., 2:4] / anchors[anchor_msk] * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
    box_loss_scale = 2 - feats[...,2:3]*feats[...,3:4]

    return raw_true_xy,raw_true_wh,raw_true_wh,box_loss_scale

def ignorer(feats,object_mask,pred_box,ignore_thresh,m):
    ignore_mask = tf.TensorArray(K.dtype(feats), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')
    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(feats[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b] , true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        return b+1, ignore_mask
    _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)
    return ignore_mask

def lossbox(object_mask,box_loss_scale,raw_true_xy,raw_pred,raw_true_wh,ignore_mask,true_class_probs,mf):
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    
    return xy_loss , wh_loss , confidence_loss , class_loss
        
def yolo_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 0.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    l_true: list of array, the output of logits
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]#yolo output
    y_true = args[num_layers:num_layers*2]
    l_true = args[num_layers*2:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    
    linput_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(l_true[0]))
    lgrid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(l_true[0])) for l in range(num_layers)]
    
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        #teacher
        object_mask = l_true[l][..., 4:5]
        true_class_probs = l_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, linput_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        
        # Darknet raw box to calculate loss.
        raw_true_xy,raw_true_wh,raw_true_wh,box_loss_scale = darknet_raw(l_true[l],object_mask,lgrid_shapes[l],grid,linput_shape,anchors,anchor_mask[l])

        # Find ignore mask, iterate over each of batch.
        ignore_mask = ignorer(l_true[l],object_mask,pred_box,ignore_thresh,m)
 

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss , wh_loss , confidence_loss , class_loss = lossbox(object_mask,box_loss_scale,raw_true_xy,raw_pred,raw_true_wh,ignore_mask,true_class_probs,mf)
        
         #alpha*t2*loss student
        loss += ( alpha * (xy_loss + wh_loss + confidence_loss + class_loss) )
        
        #########################################################################
        #student
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        
        # Darknet raw box to calculate loss.
        raw_true_xy,raw_true_wh,raw_true_wh,box_loss_scale = darknet_raw(y_true[l],object_mask,grid_shapes[l],grid,input_shape,anchors,anchor_mask[l])

        # Find ignore mask, iterate over each of batch.
        ignore_mask = ignorer(y_true[l],object_mask,pred_box,ignore_thresh,m)
 

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss , wh_loss , confidence_loss , class_loss = lossbox(object_mask,box_loss_scale,raw_true_xy,raw_pred,raw_true_wh,ignore_mask,true_class_probs,mf)
        
        #student
        # (1-alpha)*loss student
        #loss += (xy_loss + wh_loss + confidence_loss + class_loss)*1
        loss +=  ( (1-alpha) *(xy_loss + wh_loss + confidence_loss + class_loss) )
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss

def apprentice_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 1, beta = 0.5, gamma = 0.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    l_true: list of array, the output of logits
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]#yolo output
    y_true = args[num_layers:num_layers*2]
    l_true = args[num_layers*2:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    
    linput_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(l_true[0]))
    lgrid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(l_true[0])) for l in range(num_layers)]

    tinput_shape = K.cast(K.shape(l_true[0])[1:3] * 32, K.dtype(y_true[0]))
    tgrid_shapes = [K.cast(K.shape(l_true[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        
        #teacher
        grid, raw_pred, pred_xy, pred_wh = yolo_head(l_true[l],
             anchors[anchor_mask[l]], num_classes, tinput_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        
        #teacher
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        
        # Darknet raw box to calculate loss.
        raw_true_xy,raw_true_wh,raw_true_wh,box_loss_scale = darknet_raw(y_true[l],object_mask,tgrid_shapes[l],grid,input_shape,anchors,anchor_mask[l])

        # Find ignore mask, iterate over each of batch.
        ignore_mask = ignorer(y_true[l],object_mask,pred_box,ignore_thresh,m)
 

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss , wh_loss , confidence_loss , class_loss = lossbox(object_mask,box_loss_scale,raw_true_xy,raw_pred,raw_true_wh,ignore_mask,true_class_probs,mf)
        
        #teacher
        #loss += (xy_loss + wh_loss + confidence_loss + class_loss)*1
        loss += ( alpha * (xy_loss + wh_loss + confidence_loss + class_loss) )

        #########################################################################################################

        #student
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, linput_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        #student
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        
        # Darknet raw box to calculate loss.
        raw_true_xy,raw_true_wh,raw_true_wh,box_loss_scale = darknet_raw(y_true[l],object_mask,grid_shapes[l],grid,input_shape,anchors,anchor_mask[l])

        # Find ignore mask, iterate over each of batch.
        ignore_mask = ignorer(y_true[l],object_mask,pred_box,ignore_thresh,m)
 

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss , wh_loss , confidence_loss , class_loss = lossbox(object_mask,box_loss_scale,raw_true_xy,raw_pred,raw_true_wh,ignore_mask,true_class_probs,mf)
        
        #student
        #loss += (xy_loss + wh_loss + confidence_loss + class_loss)*1
        loss += ( beta * (xy_loss + wh_loss + confidence_loss + class_loss) )

        #########################################################################################################
        #apprentice
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, linput_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        #apprentice
        object_mask = l_true[l][..., 4:5]
        true_class_probs = l_true[l][..., 5:]

        # Darknet raw box to calculate loss.
        raw_true_xy,raw_true_wh,raw_true_wh,box_loss_scale = darknet_raw(l_true[l],object_mask,lgrid_shapes[l],grid,linput_shape,anchors,anchor_mask[l])

        # Find ignore mask, iterate over each of batch.
        ignore_mask = ignorer(l_true[l],object_mask,pred_box,ignore_thresh,m)
 

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss , wh_loss , confidence_loss , class_loss = lossbox(object_mask,box_loss_scale,raw_true_xy,raw_pred,raw_true_wh,ignore_mask,true_class_probs,mf)
        
        #apprentice
        loss += ( gamma *(xy_loss + wh_loss + confidence_loss + class_loss) )

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss

    
    

   
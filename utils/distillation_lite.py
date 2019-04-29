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


def tflite_out(image_data , interpreter, batch_size , num_layers=3 ,num_classes=20 ):

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
 
    input_data = image_data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    model_image_size = (416 , 416)
    fmap = model_image_size[0]//32
    
    mapsize = [1,2,4]

    outs = []
    interpreter.invoke()

    for ly in range(num_layers):
        output_data = interpreter.get_tensor(output_details[ly]['index'])
        #print(output_data.shape)
        output_data= np.reshape(output_data , (batch_size, fmap*mapsize[ly], fmap*mapsize[ly] , 3 , (num_classes + 5) ) ) 
        outs.append(output_data)


        #print(output_data.shape)
        #print(output_details)

    return outs

def data_generator_lite(annotation_lines, batch_size, input_shape, anchors, num_classes,interpreter):
    '''data generator for fit_generator'''

    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []

        #sm_pred = []
        #med_pred = []
        #lrg_pred = []
        #m_true = []

        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            
            #change data type
            image = image.astype('float32')
            #print("check")
            #print(np.expand_dims(image, axis=0).shape)
            #print(image.shape)
            image_data.append(image)
            box_data.append(box)
            
            #out_data = tflite_out( np.expand_dims(image, axis=0) )
            #print(out_data[0].shape)
            #lrg_pred.append(out_data[0])
            #med_pred.append(out_data[1])
            #sm_pred.append(out_data[2])

            i = (i+1) % n
        image_data = np.array(image_data)
        #print(image_data.shape)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        #m_true = teacher.predict(image_data)
        #print(image_data.dtype)
        #print(image_data.shape)
        #print(m_true.shape)
    
        #print( np.vstack( lrg_pred ).shape )
        #m_true.append( np.vstack( lrg_pred ) )
        #m_true.append( np.vstack( med_pred ) )
        #m_true.append( np.vstack( sm_pred ) )
        m_true =  tflite_out( image_data , interpreter  , batch_size)

        #print(m_true[0].shape )

        h, w = input_shape
        num_anchors = len(anchors)
        
        l_true =  [ np.zeros( shape=( batch_size ,416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], 9//3, 20+5) ) for l in range(3) ]

        #print(len(y_true))
        #print(len(m_true))
        #print(len(l_true))
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 

        for l in range( len(m_true) ) :

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
        
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)

def distill_data_generator_wrapper_lite(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher_path):
    #print("called")

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=teacher_path)
    
    input_details = interpreter.get_input_details()

    interpreter.resize_tensor_input(input_details[0]['index'],[ batch_size , 416, 416, 3])
    interpreter.allocate_tensors()

    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_lite(annotation_lines, batch_size, input_shape, anchors, num_classes,interpreter)
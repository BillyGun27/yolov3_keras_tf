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
                #if m_true[l][..., 4] > 0.5 :
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
        
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)

def distill_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)


def data_generator_tf(annotation_lines, batch_size, input_shape, anchors, num_classes,detection_graph,graph_names):
    '''data generator for fit_generator'''
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            tensor_output1 = sess.graph.get_tensor_by_name(graph_names[0])
            tensor_output2 = sess.graph.get_tensor_by_name(graph_names[1])
            tensor_output3 = sess.graph.get_tensor_by_name(graph_names[2])
            tensor_input = sess.graph.get_tensor_by_name(graph_names[3])
    
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
        
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                m_true = sess.run([tensor_output1,tensor_output2,tensor_output3], {tensor_input: image_data})
        
        for ly in range(len(m_true)):
            m_true[ly] = m_true[ly].reshape( m_true[ly].shape[:-1] + (3, num_classes+5 ))

        #print(m_true[0].shape)
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


def distill_data_generator_wrapper_tf(annotation_lines, batch_size, input_shape, anchors, num_classes,detection_graph,graph_names):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_tf(annotation_lines, batch_size, input_shape, anchors, num_classes,detection_graph,graph_names)

def data_generator_tf2(annotation_lines, batch_size, input_shape, anchors, num_classes,detection_graph,graph_names):
    '''data generator for fit_generator'''
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            tensor_output1 = sess.graph.get_tensor_by_name(graph_names[0])
            tensor_output2 = sess.graph.get_tensor_by_name(graph_names[1])
            tensor_output3 = sess.graph.get_tensor_by_name(graph_names[2])
            tensor_input = sess.graph.get_tensor_by_name(graph_names[3])
    
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
        
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                m_true = sess.run([tensor_output1,tensor_output2,tensor_output3], {tensor_input: image_data})
        
        for ly in range(len(m_true)):
            m_true[ly] = m_true[ly].reshape( m_true[ly].shape[:-1] + (3, num_classes+5 ))

        #print(m_true[0].shape)
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
          
            anchors_tensor = np.reshape( anchors[anchor_mask[l]] , [1, 1, 1, len( anchors[anchor_mask[l]] ) , 2] )

            grid_shape = m_true[l].shape[1:3] # height, width
            grid_shape
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                [grid_shape[0], 1, 1, 1])
            grid = np.concatenate([grid_x, grid_y],axis=-1)

            #print(l)
            m_true[l][...,:2] = y_true[l][...,:2] #(sigmoid(m_true[l][...,:2]) + grid ) / np.array( grid_shape[::-1] )
            m_true[l][..., 2:4] = y_true[l][..., 2:4] #np.exp(m_true[l][..., 2:4]) * anchors_tensor  / np.array( input_shape[::-1] )
            m_true[l][..., 4] = y_true[l][..., 4] #sigmoid(m_true[l][..., 4])
            m_true[l][..., 5:] = y_true[l][..., 5:] #sigmoid(m_true[l][..., 5:])
            
            #print("inside")
            box = np.where(y_true[l][...,4] > 0.5 )
            box = np.transpose(box)

            for i in range(len(box)):
                #if m_true[l][..., 4] > 0.5 :
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
        
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)


def distill_data_generator_wrapper_tf2(annotation_lines, batch_size, input_shape, anchors, num_classes,detection_graph,graph_names):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_tf2(annotation_lines, batch_size, input_shape, anchors, num_classes,detection_graph,graph_names)


def data_generator_weights(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher,teacher_path):
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

        #KERAS INTEFERE WITH DISTILLATION
        teacher.load_weights(teacher_path)
        for i in range(len( teacher.layers ) ): teacher.layers[i].trainable = False

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
                #if m_true[l][..., 4] > 0.5 :
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
        
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)

def distill_data_generator_wrapper_weights(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher,teacher_path):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_weights(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher,teacher_path)

def data_generator_teacher(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
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
                #if m_true[l][..., 4] > 0.5 :
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
        
        
        yield [image_data, *l_true ], np.zeros(batch_size)

def distill_data_generator_wrapper_teacher(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_teacher(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)

def data_generator_double(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
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
        '''
        m_true = teacher.predict(image_data)
         
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
        '''
        
        yield [image_data, *y_true , *y_true ], np.zeros(batch_size)

def distill_data_generator_wrapper_double(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_double(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)

def data_generator2(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
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
           
            anchors_tensor = np.reshape( anchors[anchor_mask[l]] , [1, 1, 1, len( anchors[anchor_mask[l]] ) , 2] )

            grid_shape = m_true[l].shape[1:3] # height, width
            grid_shape
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                [grid_shape[0], 1, 1, 1])
            grid = np.concatenate([grid_x, grid_y],axis=-1)

            #print(l)
            m_true[l][...,:2] = y_true[l][...,:2] #(sigmoid(m_true[l][...,:2]) + grid ) / np.array( grid_shape[::-1] )
            m_true[l][..., 2:4] = y_true[l][..., 2:4]  #np.exp(m_true[l][..., 2:4]) * anchors_tensor  / np.array( input_shape[::-1] )
            m_true[l][..., 4] = y_true[l][..., 4] #sigmoid(m_true[l][..., 4])
            m_true[l][..., 5:] = y_true[l][..., 5:] #sigmoid(m_true[l][..., 5:])
            
            #print("inside")
            box = np.where(y_true[l][...,4] > 0.5 )
            box = np.transpose(box)

            for i in range(len(box)):
                #if m_true[l][..., 4] > 0.5 :
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
        
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)

def distill_data_generator_wrapper2(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator2(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)

def data_generator_class_only(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    '''data generator for fit_generator'''
    print("class outside")

    n = len(annotation_lines)
    i = 0
    while True:
        print("class inside")

        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            #print( "i{},n{},b{},batch{}".format(i,n,b,batch_size) )
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        m_true = teacher.predict(image_data)
        print("i{},n{},b{},batch{}".format(i,n,b,batch_size))
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
            #anchors_tensor = np.reshape( anchors[anchor_mask[l]] , [1, 1, 1, len( anchors[anchor_mask[l]] ) , 2] )

            #grid_shape = m_true[l].shape[1:3] # height, width
            #grid_shape
            #grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
            #    [1, grid_shape[1], 1, 1])
            #grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
            #    [grid_shape[0], 1, 1, 1])
            #grid = np.concatenate([grid_x, grid_y],axis=-1)

            #print(l)
            #m_true[l][...,:2] = (sigmoid(m_true[l][...,:2]) + grid ) / np.array( grid_shape[::-1] )
            #m_true[l][..., 2:4] = np.exp(m_true[l][..., 2:4]) * anchors_tensor  / np.array( input_shape[::-1] )
            #m_true[l][..., 4] = sigmoid(m_true[l][..., 4])
            m_true[l][..., 5:] = sigmoid(m_true[l][..., 5:])
            
            #print("inside")
            box = np.where(y_true[l][...,4] > 0.5 )
            box = np.transpose(box)

            for i in range(len(box)):
                #if m_true[l][..., 4] > 0.5 :
                #l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
                y_true[l][ tuple(box[i]) ][5:] =  m_true[l][ tuple(box[i]) ][5:]
        
        yield [image_data, *y_true  ], np.zeros(batch_size)

def distill_data_generator_wrapper_class_only(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    print("wrapper")
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_class_only(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)


'''
def darknet_raw(feats,object_mask,grid_shape,grid,input_shape,anchors,anchor_msk):
    raw_true_xy = feats[..., :2]*grid_shape[::-1] - grid
    raw_true_wh = K.log(feats[..., 2:4] / anchors[anchor_msk] * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
    box_loss_scale = 2 - feats[...,2:3]*feats[...,3:4]

    return raw_true_xy,raw_true_wh,box_loss_scale

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


def old_yolo_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 0.5, print_loss=False):
    Return yolo_loss tensor

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
        raw_true_xy,raw_true_wh,box_loss_scale = darknet_raw(l_true[l],object_mask,lgrid_shapes[l],grid,linput_shape,anchors,anchor_mask[l])

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
        raw_true_xy,raw_true_wh,box_loss_scale = darknet_raw(y_true[l],object_mask,grid_shapes[l],grid,input_shape,anchors,anchor_mask[l])

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
'''
def teacher_yolo_loss(yolo_outputs,y_true,anchors,num_classes , ignore_thresh ,input_shape,grid_shapes,m,mf):
    
    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs,
        anchors, num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    _ , box_wh, box_confidence, box_class_probs = yolo_head(y_true, anchors, num_classes, input_shape, calc_loss=False)

    object_mask = box_confidence #y_true[..., 4:5]
    true_class_probs = box_class_probs #y_true[..., 5:]

    raw_true_xy = y_true[..., :2]
    raw_true_wh =y_true[..., 2:4]
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
    box_loss_scale = 2 - true_wh[0] *true_wh[1]


    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')

    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        return b+1, ignore_mask 
    _, ignore_mask  = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)

    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf

    return xy_loss , wh_loss , confidence_loss ,class_loss ,ignore_mask

def basic_yolo_loss(yolo_outputs,y_true,anchors,num_classes , ignore_thresh ,input_shape,grid_shapes,m,mf):
    object_mask = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs,
        anchors, num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = y_true[..., :2]*grid_shapes[::-1] - grid
    raw_true_wh = K.log(y_true[..., 2:4] / anchors * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
    box_loss_scale = 2 - y_true[...,2:3]*y_true[...,3:4]

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')

    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        return b+1, ignore_mask 
    _, ignore_mask  = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)

    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf

    return xy_loss , wh_loss , confidence_loss ,class_loss ,ignore_mask


def yolo_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 0, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes 
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
    

    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        #teacher
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask  = basic_yolo_loss(yolo_outputs[l],l_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss += ( alpha * (xy_loss + wh_loss + confidence_loss + class_loss) )

        #loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss teacher: ')

        #student
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask = basic_yolo_loss(yolo_outputs[l],y_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss += ( (1-alpha) * (xy_loss + wh_loss + confidence_loss + class_loss) )

        #loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss student: ')

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss

def joint_yolo_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 0, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes 
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
    

    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        #teacher
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask  = teacher_yolo_loss(yolo_outputs[l],l_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss += ( alpha * (xy_loss + wh_loss + confidence_loss + class_loss) )

        #loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss teacher: ')

        #student
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask = basic_yolo_loss(yolo_outputs[l],y_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss += ( (1-alpha) * (xy_loss + wh_loss + confidence_loss + class_loss) )

        #loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss student: ')

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss

def noalpha_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 0, print_loss=True):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes 
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
    

    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        #teacher
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask  = basic_yolo_loss(yolo_outputs[l],l_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss +=  (xy_loss + wh_loss + confidence_loss + class_loss) 

        #loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss teacher: ')

        #student
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask = basic_yolo_loss(yolo_outputs[l],y_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss +=  (xy_loss + wh_loss + confidence_loss + class_loss) 

        #loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss student: ')

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss

def fake_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 0, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes 
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
    

    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        #teacher
        #xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask  = basic_yolo_loss(yolo_outputs[l],l_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        #loss += ( alpha * (xy_loss + wh_loss + confidence_loss + class_loss) )

        #loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss teacher: ')

        #student
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask = basic_yolo_loss(yolo_outputs[l],y_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        #loss += ( (1-alpha) * (xy_loss + wh_loss + confidence_loss + class_loss) )
        loss += xy_loss + wh_loss + confidence_loss + class_loss

        #loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss student: ')

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss


def basic_apprentice_loss(yolo_outputs,y_true,anchors,num_classes , ignore_thresh ,input_shape,grid_shapes,m,mf):
    # student -> raw to raw and normal xywh  , teacher -> raw to raw and normal wh
 
    object_mask = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs,
        anchors, num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = y_true[..., :2]
    raw_true_wh =y_true[..., 2:4]
    _, _, _, true_wh = yolo_head(y_true, anchors, num_classes, input_shape, calc_loss=True)
    box_loss_scale = 2 - true_wh[0] *true_wh[1]

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')

    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        return b+1, ignore_mask 
    _, ignore_mask  = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)

    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf

    return xy_loss , wh_loss , confidence_loss ,class_loss ,ignore_mask

def apprentice_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 1, beta = 0.5, gamma = 0.5 , print_loss=True):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes 
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
    

    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
       
        #teacher  (teacher , label)
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask  = basic_yolo_loss(l_true[l],y_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss += ( alpha * (xy_loss + wh_loss + confidence_loss + class_loss) )

        loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss teacher: ')

        #student (student , label)
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask = basic_yolo_loss(yolo_outputs[l],y_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss += (  beta * (xy_loss + wh_loss + confidence_loss + class_loss) )

        loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss student: ')

        #apprentice  (student , teacher)
        xy_loss , wh_loss , confidence_loss ,class_loss , ignore_mask  = basic_apprentice_loss(yolo_outputs[l],l_true[l], anchors[anchor_mask[l]], num_classes , ignore_thresh ,input_shape,grid_shapes[l],m,mf)
        loss += ( gamma * (xy_loss + wh_loss + confidence_loss + class_loss) )

        loss = tf.Print(loss, [l, loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss apprentice: ')

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss

def old_apprentice_distill_loss(args, anchors, num_classes, ignore_thresh=.5, alpha = 1, beta = 0.5, gamma = 0.5, print_loss=False):
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
        '''
        #teacher
        grid, raw_pred, pred_xy, pred_wh = yolo_head(l_true[l],
             anchors[anchor_mask[l]], num_classes, tinput_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        
        #teacher
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        
        # Darknet raw box to calculate loss.
        raw_true_xy,raw_true_wh,box_loss_scale = darknet_raw(y_true[l],object_mask,tgrid_shapes[l],grid,input_shape,anchors,anchor_mask[l])

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
        raw_true_xy,raw_true_wh,box_loss_scale = darknet_raw(y_true[l],object_mask,grid_shapes[l],grid,input_shape,anchors,anchor_mask[l])

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
        raw_true_xy,raw_true_wh,box_loss_scale = darknet_raw(l_true[l],object_mask,lgrid_shapes[l],grid,linput_shape,anchors,anchor_mask[l])

        # Find ignore mask, iterate over each of batch.
        ignore_mask = ignorer(l_true[l],object_mask,pred_box,ignore_thresh,m)
 

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss , wh_loss , confidence_loss , class_loss = lossbox(object_mask,box_loss_scale,raw_true_xy,raw_pred,raw_true_wh,ignore_mask,true_class_probs,mf)
        
        #apprentice
        loss += ( gamma *(xy_loss + wh_loss + confidence_loss + class_loss) )
        '''
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message=' loss: ')
    return loss

    
    

   
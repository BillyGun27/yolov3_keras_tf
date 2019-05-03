"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 
from keras.layers import Input, Lambda ,Reshape

from utils.core import preprocess_true_boxes
from utils.utils  import get_random_data
from utils.train_tool import get_classes,get_anchors
from utils.evaluation import AveragePrecision

from utils.train_tool import get_classes,get_anchors,data_generator_wrapper
from utils.distillation import distill_data_generator_wrapper2 #distill_data_generator_wrapper_class_only ,distill_data_generator_wrapper_weights

#changeable param
from utils.core import yolo_loss as yolo_custom_loss
#from model.mobilenet import yolo_body
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

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,load_pretrained=False,
            freeze_body=2, weights_path=model_path ) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
     
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

    meanAP = AveragePrecision(data_generator_wrapper(val_lines[:200], 1 , input_shape, anchors, num_classes) , 200 , input_shape , len(anchors)//3 , anchors ,num_classes,log_dir)

    #declare model
    num_anchors = len(anchors)
    image_input = Input(shape=(416, 416, 3))
    print("load model")
    teacher = teacher_body(image_input, num_anchors//3, num_classes)
    print("load modal end")
    print("load weight")
    teacher.load_weights(teacher_path)
    print("load weight end")

    print("load reshape")
    # return the constructed network architecture
    # class+5
    yolo3 = Reshape((13, 13, 3, 25))(teacher.layers[-3].output)
    yolo2 = Reshape((26, 26, 3, 25))(teacher.layers[-2].output)
    yolo1 = Reshape((52, 52, 3, 25))(teacher.layers[-1].output)
    print("load reshape end")

    print("construct model")
    teacher = Model( inputs= teacher.input , outputs=[yolo3,yolo2,yolo1] )
    print("construct model end")

    print("load freeze and predict end")
    for i in range(len( teacher.layers ) ): teacher.layers[i].trainable = False
    #teacher._make_predict_function()
    print("load freeze and predict end")


    #print ( teacher.layers[-3].get_weights() )
    #print ( teacher.layers[-2].get_weights() )
    #print ( teacher.layers[-1].get_weights() )
    #print ( len( teacher.layers[-6].get_weights() ) )
    #print ( teacher.layers[-6].get_weights()[0].shape )
    #print ( teacher.layers[-6].get_weights()[1].shape )
    #print ( teacher.layers[-6].get_weights()[1][0] )
    #print ( teacher.layers[-6].get_weights()[1][0] )
    #print ( teacher.layers[-6].get_weights()[0][0][0][0][0] )
    #print ( teacher.layers[-6].get_weights()[0][0][0][0][0] )
    
    #print ( teacher.layers[-6].get_weights()[0] )
    #print ( teacher.layers[-5].get_weights() )
    #print ( teacher.layers[-4].get_weights() )

    #teacher.summary()
    #print(len(teacher.layers))
    
    batch_size = 1
    datagen =  distill_data_generator_wrapper2(train_lines, batch_size, input_shape, anchors, num_classes,teacher)
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
    #print(logits[4])
    arrp = logits[4]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print(logits[4][tuple(box[0])])
    
    
    '''
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
             'yolo_custom_loss' : lambda y_true, y_pred: y_pred})

        batch_size = 2#32

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(distill_data_generator_wrapper_class_only(train_lines, batch_size, input_shape, anchors, num_classes,teacher),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=distill_data_generator_wrapper_class_only(train_lines, batch_size, input_shape, anchors, num_classes,teacher),
                validation_steps=max(1, num_val//batch_size),
                epochs=epoch_end_first,
                initial_epoch=0,
                callbacks=[logging, checkpoint])

        last_loss = history.history['loss'][-1]
        last_val_loss = history.history['val_loss'][-1]

        hist = "loss{0:.4f}-val_loss{1:.4f}".format(last_loss,last_val_loss)

        model.save_weights(log_dir + "last_"+ hist + ".h5")

        model.save_weights(log_dir + model_name+'_trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={ 
            'yolo_custom_loss' : lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size =  1#32 note that more GPU memory is required after unfreezing the body
        

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(distill_data_generator_wrapper_class_only(train_lines, batch_size, input_shape, anchors, num_classes,teacher),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=distill_data_generator_wrapper_class_only(val_lines, batch_size, input_shape, anchors, num_classes,teacher),
            validation_steps=max(1, num_val//batch_size),
            epochs=epoch_end_final,
            initial_epoch=epoch_end_first,
            callbacks=[logging, checkpoint, reduce_lr])#, early_stopping
       
        last_loss = history.history['loss'][-1]
        last_val_loss = history.history['val_loss'][-1]

        hist = "loss{0:.4f}-val_loss{0:.4f}".format(last_loss,last_val_loss)

        model.save_weights(log_dir + "last_"+ hist + ".h5")

        model.save_weights(log_dir + model_name + '_trained_weights_final.h5')
    
    # Further training if needed.
    '''
def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    #l_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
    #    num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    
    for y in range(-3, 0):
        model_body.layers[y].name = "conv2d_output_" + str(h//{-3:32, -2:16, -1:8}[y])

    model_loss = Lambda(yolo_custom_loss, output_shape=(1,), name='yolo_custom_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5  })(
        [*model_body.output, *y_true ])
    model = Model([model_body.input, *y_true  ], model_loss)


    return model



if __name__ == '__main__':
    _main()

'''
    batch_size = 1
    datagen =  distill_data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes,teacher)
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
    #print(logits[4])
    arrp = logits[4]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print( logits[4][tuple(box[0])] )

    print(logits[2].shape)
    #print(logits[1])
    arrp = logits[2]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print(logits[2][tuple(box[0])])

    print(logits[5].shape)
    #print(logits[4])
    arrp = logits[5]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print( logits[5][tuple(box[0])] )

    print(logits[3].shape)
    #print(logits[1])
    arrp = logits[3]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print(logits[3][tuple(box[0])])

    print(logits[6].shape)
    #print(logits[4])
    arrp = logits[6]
    box = np.where(arrp[...,4] > 0 )
    box = np.transpose(box)
    print(box)
    if( len(box) ):
        print( logits[6][tuple(box[0])] )
'''

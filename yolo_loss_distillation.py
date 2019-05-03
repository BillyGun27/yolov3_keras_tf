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

#changeable param
from utils.distillation import DistillCheckpointCallback 
from utils.distillation import yolo_distill_loss as yolo_custom_loss
from model.small_mobilenets2 import yolo_body
from model.yolo3 import yolo_body as teacher_body

#from keras.utils.vis_utils import plot_model as plot

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import argparse

def _main():
    epoch_end_first = 30
    epoch_end_final = 60
    model_name = 'joint_mobilenet'
    log_dir = 'logs_d/loss_apprentice_joint_mobilenet_000/'
    model_path = 'model_data/fake_trained_weights_final_mobilenet.h5'
    #teacher_path ="logs/new_yolo_000/last_loss16.9831-val_loss16.9831.h5"
    teacher_path = "model_data/trained_weights_final.h5"

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
        model , student , teacher = create_model(input_shape, anchors, num_classes,  load_pretrained=False,
            freeze_body=2, weights_path=model_path , teacher_weights_path=teacher_path ) # make sure you know what you freeze


    logging = TensorBoard(log_dir=log_dir)
    checkpointStudent = DistillCheckpointCallback(model, model_name , log_dir)
    #checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
     
    with open(train_path) as f:
        train_lines = f.readlines()
    #train_lines = train_lines[:8]

    with open(val_path) as f:
        val_lines = f.readlines()
    #val_lines = val_lines[:8]
   # with open(test_path) as f:
   #     test_lines = f.readlines()

    num_train = int(len(train_lines))
    num_val = int(len(val_lines))

    meanAP = AveragePrecision(data_generator_wrapper(val_lines, 1 , input_shape, anchors, num_classes) , num_val , input_shape , len(anchors)//3 , anchors ,num_classes,log_dir)

    
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if False:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
             'yolo_custom_loss' : lambda y_true, y_pred: y_pred})

        batch_size = 4#24#32

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=epoch_end_first,
                initial_epoch=0,
                callbacks=[logging, checkpointStudent ])

        last_loss = history.history['loss'][-1]
        last_val_loss = history.history['val_loss'][-1]

        hist = "loss{0:.4f}-val_loss{1:.4f}".format(last_loss,last_val_loss)

        model.save( hist + "model_checkpoint.h5")
        student.save_weights(log_dir + "last_"+ hist + ".h5")
        student.save_weights(log_dir + model_name + '_trained_weights_final.h5')
        teacher.save_weights(log_dir + "teacher" + model_name + '_trained_weights_final.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if False:
        for i in range(len(student.layers)):
            student.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={ 
            'yolo_custom_loss' : lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size =  4#32 note that more GPU memory is required after unfreezing the body

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epoch_end_final,
            initial_epoch=epoch_end_first,
            callbacks=[logging, reduce_lr ,checkpointStudent ]) #, early_stopping
        model.save_weights(log_dir + model_name + '_trained_weights_final.h5')

        last_loss = history.history['loss'][-1]
        last_val_loss = history.history['val_loss'][-1]

        hist = "loss{0:.4f}-val_loss{0:.4f}".format(last_loss,last_val_loss)

        student.save_weights(log_dir + "last_"+ hist + ".h5")
        student.save_weights(log_dir + model_name + '_trained_weights_final.h5')
        teacher.save_weights(log_dir + "teacher" + model_name + '_trained_weights_final.h5')
       
    
    # Further training if needed.

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5',teacher_weights_path="model_data/trained_weights_final.h5"):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
    
    teacher = teacher_body(image_input, num_anchors//3, num_classes)
    teacher.load_weights(teacher_weights_path)

    #teacher_output = []
    #for ty in range(-3, 0):#13,26,52
    #    out_ty = Reshape( teacher.layers[ty].output_shape[:-1] + ( 3, num_classes+5 )  )(teacher.layers[ty].output)
        
    #yolo13 = Reshape( ( teacher.layers[-3].output_shape[:-1] + ( 3, num_classes+5 ) ) )(teacher.layers[-3].output)
    #yolo26 = Reshape( ( teacher.layers[-3].output_shape[:-1] + ( 3, num_classes+5 ) ) )(teacher.layers[-2].output)
    #yolo52 = Reshape( ( teacher.layers[-3].output_shape[:-1] + (3, num_classes+5 ) ) )(teacher.layers[-1].output)
    
    #teacher = Model( inputs= teacher.input , outputs= teacher_output )
    #print("teacher before freeze" , len(teacher.trainable_weights) )
    for i in range(len( teacher.layers ) ): teacher.layers[i].trainable = False
    #print("teacher after freeze" , len(teacher.trainable_weights) )

    student = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    #print("student before freeze" ,  len(student.trainable_weights) )
    
    if load_pretrained:
        student.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(student.layers)-3)[freeze_body-1]
            for i in range(num): student.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(student.layers)))
    
    for y in range(-3, 0):
        student.layers[y].name = "conv2d_output_" + str(h//{-3:32, -2:16, -1:8}[y])


    model_loss = Lambda(yolo_custom_loss, output_shape=(1,), name='yolo_custom_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*student.output, *y_true , *teacher.output  ])
    model = Model([ image_input , *y_true  ], model_loss)

    #print("student after freeze" ,  len(student.trainable_weights) )
    #print("model" , len( model.trainable_weights) )

    #for i in range(len( student.layers ) ): student.layers[i].trainable = True

    #print("student after unfreeze" ,  len(student.trainable_weights) )

    #print("model after" ,  len(model.trainable_weights) )

    #from keras.utils.vis_utils import plot_model as plot
    #plot(model, to_file='{}.png'.format("train_together"), show_shapes=True)
    #print("stop")

    return model , student , teacher




if __name__ == '__main__':
    _main()

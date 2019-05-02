# # load & inference the model ==================

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda ,Reshape , Convolution2D, MaxPooling2D, Activation
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from utils.utils import letterbox_image


from tensorflow.python.platform import gfile
from model.small_mobilenets2 import yolo_body
from tensorflow import Graph


img = "test_data/london.jpg"
image = Image.open(img)

#image_shape = ( image.size[1], image.size[0] , 3)
model_image_size = (416 , 416)

model_image_size[0]%32 == 0, 'Multiples of 32 required'
model_image_size[1]%32 == 0, 'Multiples of 32 required'
boxed_image = letterbox_image(image,tuple(reversed(model_image_size)))

image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)

print(image_data.shape)

model_path = 'model_data/small_mobilenets2_trained_weights_final.h5'

image_input = Input(shape=(416, 416, 3))

'''
class Model:
    def __init__(self, path):
       self.model =  yolo_body(image_input, 3, 20)
       self.model.load_weights(path)
       self.graph = tf.get_default_graph()

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)

model1 = Model(model_path)
p = model1.predict(image_data)
print(p)
'''

modelA = yolo_body(image_input, 3, 20)
modelA.load_weights(model_path)
AA = modelA.predict(image_data)
print(AA[0].shape)

inp = Input(shape=(13,13,75))
x = Convolution2D(10, (3, 3), padding='same')(inp)
x = Activation('relu')(x)
mod = Model(inp, x)

s = mod.predict(AA[0])
print(s)
'''
from tensorflow import Graph
from keras.models import model_from_json
print('loading first model...')
graph1 = Graph()
with graph1.as_default():
    modelA = yolo_body(image_input, 3, 20)
    modelA.load_weights(model_path)

print('loading second model...')
graph2 = Graph()
with graph2.as_default():
    modelB = yolo_body(image_input, 3, 20)
    modelB.load_weights(model_path) 
'''

#modelA = yolo_body(image_input, 3, 20)
#modelA.load_weights(model_path)
#AA = modelA.predict(image_data)
#print(AA[0].shape)
#print(AA[0][0,6,6,1])

#modelB = yolo_body(image_input, 3, 20)
#modelB.load_weights(model_path)
#BB = modelB.predict(image_data)
#print(BB[0].shape)
#print(BB[0][0,6,6,1])


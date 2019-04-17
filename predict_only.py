import numpy as np
from keras.models import Model,load_model
from PIL import Image, ImageFont, ImageDraw
from utils.utils import letterbox_image
from utils.train_tool import get_classes,get_anchors
from timeit import default_timer as timer

img = "test_data/london.jpg"
image = Image.open(img)
model_image_size = (224 , 224)

image_shape = ( image.size[1], image.size[0] , 3)

model_image_size[0]%32 == 0, 'Multiples of 32 required'
model_image_size[1]%32 == 0, 'Multiples of 32 required'
boxed_image = letterbox_image(image,tuple(reversed(model_image_size)))

image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)

#model = load_model('model_data/1scale_small_mobilenet_trained_model.h5' , compile=False )
#model = load_model('model_data/2scale_small_mobilenet_trained_model.h5' , compile=False )
#model = load_model('model_data/tiny_yolo.h5' , compile=False )
model = load_model('model_data/1scale_tiny_yolo_model.h5' , compile=False )
#model = load_model('model_data/small_mobilenet_trained_model.h5' , compile=False )

print('start')
start = timer()
test = model.predict(image_data)
end = timer()
print(end - start)

start = timer()
test = model.predict(image_data)
end = timer()
print(end - start)
#print(start)
#print(end)

#print(test)
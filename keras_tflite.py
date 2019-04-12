from tensorflow.contrib import lite
converter = lite.TFLiteConverter.from_keras_model_file( 'model.h5' ) # Your model's name
model = converter.convert()
file = open( 'model.tflite' , 'wb' ) 
file.write( model )
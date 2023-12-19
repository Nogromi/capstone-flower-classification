import tensorflow as tf
from tensorflow import keras

def convert_model(keras_model_name, tflite_model_name):
    model = keras.models.load_model(keras_model_name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print('... saving TF-lite model to ./models')
    with open(tflite_model_name, 'wb') as f_out:
        f_out.write(tflite_model)

#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

classes=['california_poppy',
 'water_lily',
 'astilbe',
 'iris',
 'coreopsis',
 'tulip',
 'black_eyed_susan',
 'daffodil',
 'calendula',
 'common_daisy',
 'magnolia',
 'sunflower',
 'rose',
 'carnation',
 'dandelion',
 'bellflower']

# remove dependency of tensorflow
interpreter = tflite.Interpreter(model_path='models/flower-model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# remove dependency for preprocessing
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 255
    # x -= 1.
    return x

def read_img(url):
    img=download_image(url)
    img=prepare_image(img,(128,128))
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)
    return X

def predict(url):
    X = read_img(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    result_probability = preds[0].tolist()
    return result_probability


def get_class(url):
    pred=predict(url)
    return dict(zip(classes, pred))

def lambda_handler(event, context):
    url = event['url']
    result = get_class(url)
    return result


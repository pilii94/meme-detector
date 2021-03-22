import os
from PIL import Image
import numpy as np
import io

import urllib.request
import flask
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
from os import makedirs
from os.path import exists, join

import tensorflow as tf
from tensorflow import keras


print("Tensorflow Version ",tf.__version__)


## Initialise Flask
app = flask.Flask(__name__)


## load model
@app.before_first_request
def load_model_keras_model():
    global model
    model = keras.models.load_model('./my_model.h5')
    print("=============================Model Loaded==========================")

global CATEGORIES
CATEGORIES = ['Memes', 'Not Memes']

# Maximum Image Uploading size
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

# Image extension allowed
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


print("Loading keras model")
load_model_keras_model()


## ping server request
@app.route('/pingserver',methods=['GET'])
def pingServer():

    pingResp = {'response_message': "Pong from server", 'response_code': "200", "response_root": "Success"}

    return jsonify(pingResp)

## For multiple files upload..
@app.route('/predict', methods=['POST'])
def predict():

    responseFileList = []

    if 'file' not in request.files:
        fileResp = {'response_message': "No file part in the request", 'response_code': "400", "response_root": "Error"}
        responseFileList.append(fileResp)

    # uploaded_files = request.files.getlist("file")

    f = request.files['file']
    sfname = 'static/'+str(secure_filename(f.filename))
    f.save(sfname)


    try:

        if f.filename == '':
            fileResp = {'response_message': "No file part in the request", 'response_code': "400",
                        "response_root": "Error"}
            fileResp = {'file': filename}
            responseFileList.append(fileResp)

        # Check if the file is one of the allowed types/extensions
        elif f and allowed_file(f.filename):
            filename = secure_filename(f.filename)

            class_label = predict_label(sfname)


            if (class_label == ''):

                fileResp = {'response_message': "Something went wrong", 'response_code': "500",
                            "response_root": "Error", 'imagePath': filename, 'memeStatus': "Not Defined"}
                responseFileList.append(fileResp)
            else:

            	## Success Response
                fileResp = {'response_message': "Valid File", 'response_code': "200",
                            "response_root": "Success", 'imagePath': filename, 'memeStatus': class_label}
                responseFileList.append(fileResp)

        else:
            fileResp = {'response_message': "incompatible file extension part in the request",
                        'response_code': "400",
                        "response_root": "Error"}
            responseFileList.append(fileResp)

    except Exception as e:  # in the interest in keeping the output clean...
        filename = secure_filename(f.filename)
        fileResp = {'response_message': "File not uploaded", 'response_code': "500",
                    "response_root": "Error", 'file': filename}
        responseFileList.append(fileResp)
        print(e)
        pass

    return jsonify(responseFileList)

######################################################

def predict_label(f):

    class_label = ''
   
    img_height = 128
    img_width = 128

    try:
        image = tf.keras.preprocessing.image.load_img(f,target_size =(img_height, img_width))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) 
        predictions = model.predict_classes(input_arr)
        class_label = CATEGORIES[predictions[0]]

    except Exception as e:
        pass
        print("Exception---->", e)

    return class_label


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    # app.run(host='34.93.214.159', port=5001, debug=True, threaded=True)
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)




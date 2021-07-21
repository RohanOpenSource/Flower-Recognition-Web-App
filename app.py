from flask import Flask, render_template, request, flash, redirect, url_for
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

app.secret_key = "secret key"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation="softmax"))

model.load_weights('FlowerModel.h5')

class_names=['Dandelion', 'Daisy', 'Tulip', 'Sunflower', 'Rose']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return render_template('app.html')

@app.route("/", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        img = Image.open(request.files['file'])
        img = tf.image.resize(img, (224, 224))/255.0
        img = img.numpy()
        img = img.reshape(1, 224, 224, 3)
        number=np.argmax(model(img)[0])
        name=class_names[number]
        flash('Image successfully uploaded and displayed below')
        return render_template('app.html', variable=name)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)


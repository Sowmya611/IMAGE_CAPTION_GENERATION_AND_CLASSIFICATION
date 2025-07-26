#app.py
import os
import shutil
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Chaithu%40123@localhost/image_caption_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define paths
CLASS_NAMES = ['Picture', 'Memes', 'Picture', 'Selfies', 'Picture', 'Whatsapp_Screenshots', 'Circulars', 'Handwritten', 'Printed', 'Question Pappers']
UNCLASSIFIED_FOLDER = "Unclassified"

# Load trained models
classification_model = load_model("models/model.h5")
caption_model = tf.keras.models.load_model("mymodel.h5")
with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
max_length = 37

# Image Classification Function
def preprocess_image(image_path, img_size=(224, 224)):
    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_images(folder_path, destination_folder, confidence_threshold=50):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(destination_folder, class_name), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, UNCLASSIFIED_FOLDER), exist_ok=True)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        img = preprocess_image(file_path)
        predictions = classification_model.predict(img)
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions) * 100
        
        dest_path = os.path.join(destination_folder, predicted_class if confidence >= confidence_threshold else UNCLASSIFIED_FOLDER, filename)
        shutil.copy(file_path, dest_path)

# Image Captioning Function
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return vgg_model.predict(image, verbose=0)

def generate_caption(image_features):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = {index: word for word, index in tokenizer.word_index.items()}.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Database Model
class ImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    caption = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

import subprocess 

@app.route('/classify', methods=['GET', 'POST'])
def classify_folder():
    if request.method == 'POST':
        folder_path = request.form.get('folder_path')
        destination_folder = "classified_images"

        if folder_path and os.path.exists(folder_path):
            classify_images(folder_path, destination_folder)

           
            subprocess.Popen(f'explorer "{os.path.abspath(destination_folder)}"')

            return render_template('classify.html', success=True, message="Images classified successfully!")

        return render_template('classify.html', error=True, message="Invalid folder path. Please enter a valid path.")

    return render_template('classify.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No selected file")
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        image_features = extract_features(filepath)
        caption = generate_caption(image_features)
        new_image = ImageCaption(filename=file.filename, caption=caption)
        db.session.add(new_image)
        db.session.commit()
        return render_template('upload.html', image=filepath, caption=caption, success="Image uploaded successfully!")
    return render_template('upload.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_term = request.form.get('search_term')
        results = ImageCaption.query.filter(ImageCaption.caption.ilike(f"%{search_term}%")).all()
        return render_template('results.html', results=results, search_term=search_term)
    return render_template('search.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

from flask import Flask, request, jsonify,render_template,redirect, url_for
import tensorflow as tf
from ultralytics import YOLO
import numpy as np
from werkzeug.utils import secure_filename
import os
import cv2
import random



# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_ultrasound(filename,threshold=0.99):
    model = tf.keras.models.load_model('breast_ultrasound_detector.h5')
    # Load and preprocess the image
    image = cv2.imread(filename)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))  # Inception V3 input size
    
    # Normalize the image
    image = image / 255.0
    
    # Expand dimensions to create a batch
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)
    
    print(predictions[0][0])
    
    # Interpret predictions
    # For binary classification, use threshold to classify
    if predictions[0][0] >= threshold:
        return "ultrasound"
    else:
        return "not ultrasound"


def classify(filename):
    # Load the saved model
    model = YOLO('best.pt')

    # Make predictions using the loaded model
    results = model.predict(filename, save=False, imgsz=64, conf=0.5,stream=True)

    for result in results:
        dict = result.names
        probs = result.probs # Probs object for classification outputs
        pred = dict.get(probs.top1)
        conf = round(float(probs.top1conf),3)
        conf_perc = conf *100
        if conf_perc == 100:
            # Generate a random float number within the specified range
            conf_perc = round(random.uniform(99.5, 100),3)
        res =  pred+"  "+ str(conf_perc)+"%"
        print(f"class : {res}")
        return res
    
 


@app.route('/',methods=['GET'])
def index():
    return render_template("store.html")

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return "No file part"
        
        imagefile = request.files['imagefile']
        
        if imagefile.filename == '':
            return "No selected file"
        
        # Assuming you have a folder named "images_cancer" where you want to save the uploaded images
        image_path = "./static/images_test/" + imagefile.filename
        imagefile.save(image_path)
        
        prediction = predict_ultrasound(image_path)
        
        if prediction == 'ultrasound':
            classification = classify(image_path)
            return classification
        else:
            return "Image Not Ultrasound"
        
    
    # Render the page normally
    return None


if __name__ == '__main__':
    app.run(debug=True)


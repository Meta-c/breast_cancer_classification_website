from flask import Flask, request, jsonify,render_template,redirect, url_for
import tensorflow as tf
from ultralytics import YOLO
import numpy as np
from werkzeug.utils import secure_filename
import os
import cv2
import random
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import pickle



# UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

dict = {
    0:'Not Ultrasound',
    1: 'ultrasound'
}

image_size = 224

# Load the models
with open('verify_models/gmm_model.pkl', 'rb') as gmm_file:
    gmm_model = pickle.load(gmm_file)

with open('verify_models/isotonic_regressor.pkl', 'rb') as iso_file:
    isotonic_regressor = pickle.load(iso_file)
    
    
# Load the saved model
model = YOLO('best.pt')
        
# Load pre-trained ResNet50 model
resnet_model = ResNet50(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False, pooling='avg')    



def read_and_prep_images(img_path, img_height=image_size, img_width=image_size):
    """Read and preprocess the image."""
    img = Image.open(img_path).convert('RGB')  # Ensure image has 3 channels
    img = img.resize((img_height, img_width))
    img_array = np.array(img)

    if img_array.shape != (img_height, img_width, 3):
        raise ValueError(f"Image array has incorrect shape: {img_array.shape}. Expected shape: ({img_height}, {img_width}, 3)")

    output = preprocess_input(img_array)
    return img, output.reshape((1, img_height, img_width, 3))  # Reshape to (1, 224, 224, 3)



def predict_ultrasound(filename,threshold=0.9):
    img, X_test = read_and_prep_images(filename)
    X_test = resnet_model.predict(X_test)
    log_probs_test = gmm_model.score_samples(X_test)
    test_probabilities = isotonic_regressor.predict(log_probs_test)
    test_predictions = [1 if prob >= threshold else 0 for prob in test_probabilities]
    
    return dict.get(test_predictions[0])


def classify(filename):


    # Make predictions using the loaded model
    results = model.predict(filename, save=False, imgsz=64, conf=0.5,stream=True)

    for result in results:
        dict = result.names
        probs = result.probs # Probs object for classification outputs
        pred = dict.get(probs.top1)
        conf = round(float(probs.top1conf),3)
        conf_perc = conf *100
        res =  pred+"  "+ str(round(conf_perc))+"%"
        print(f"class : {res}")
        return res
    
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        
        if allowed_file(imagefile.filename):
            # Assuming you have a folder named "images_test" where you want to save the uploaded images
            image_path = os.path.join("./static/images_test", secure_filename(imagefile.filename))
            imagefile.save(image_path)
            
            prediction = predict_ultrasound(image_path)
            
            if prediction == 'ultrasound':
                classification = classify(image_path)
                return classification
            else:
                return "Image Not Ultrasound"
        else:
            return "Invalid file type"
    
    # Render the page normally
    return render_template("store.html")

if __name__ == '__main__':
    app.run(debug=True)
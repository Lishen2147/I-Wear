from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2 as cv
import warnings
import pickle

app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore")

# Load the model

# Create processing function w/ PIL for image

@app.route('/predict', methods=['POST'])
def predict():
    print("Request received")
    if 'uploadFile' not in request.files:
        return jsonify({'prediction': 'null', 'error': 'No file provided'})
    
    image = request.files['uploadFile']
    print(type(image))

    # handle processing & predicting here
    # prediction = model.predict(image)
    # return jsonify({'prediction': prediction.tolist()})

    print("Request processed")
    return jsonify({'prediction': 'success', 'error': 'null'})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'})
    image = request.files['image']
    print(type(image))
    # prediction = model.predict(image)
    # return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
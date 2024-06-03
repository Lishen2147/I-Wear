from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import pickle
import base64
import json
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.CNN_app import predict_face_shape
from glasses_tryon.with_image import generate_images

constants_file = open("../constants.json")
model_constants = json.load(constants_file)["model_params"]
MODEL_PATH = "../model/best_model.pth"

app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore")


@app.route('/predict', methods=['POST'])
def predict(): 
    data = request.get_json()
    
    if data is None or 'image' not in data:
        return jsonify({'prediction': 'failed', 'error': 'No image data provided'}), 400
    try:
        base64_image = data['image']

        if not base64_image.startswith("data:image"):
            return jsonify({'prediction': 'failed', 'error': 'Invalid image data'}), 400

        base64_data = base64_image.split(',')[1]

        image_data = base64.b64decode(base64_data)

        image_type = "jpg" if "jpeg" in base64_image.lower() else "png"
        file_path = f'./uploaded_image.{image_type}'

        with open(file_path, 'wb') as f:
            f.write(image_data)
        _, predicted_shapes_with_probabilities = predict_face_shape("./uploaded_image.jpg", MODEL_PATH, model_constants["num_classes"])
        predict_structure = model_constants["facial_structure_classes"][np.argmax([x[1] for x in predicted_shapes_with_probabilities])]
        return jsonify({'prediction': predict_structure, 'prediction_probabilities': predicted_shapes_with_probabilities, 'error': 'null', 'response_code': 200}), 200
    except:
        return jsonify({'prediction': None, 'prediction_probabilities': None, 'error': 'null', 'response_code': 400}), 400

@app.route('/glasses', methods=['POST'])
def glasses():
    data = request.get_json()
    if data is None or 'image' not in data or 'prediction' not in data:
        return jsonify({'prediction': 'failed', 'error': 'No image data provided'}), 400
    try:
        base64_image = data['image']
        prediction = data['prediction']
        
        if not base64_image.startswith("data:image"):
            return jsonify({'prediction': 'failed', 'error': 'Invalid image data'}), 400
        base64_data = base64_image.split(',')[1]
        generated_images = generate_images(prediction, base64_data)
        
        return jsonify({'images': generated_images, 'error': 'null', 'response_code': 200}), 200
    except:
        return jsonify({'prediction': None, 'prediction_probabilities': None, 'error': 'null', 'response_code': 400}), 400
if __name__ == '__main__':
    app.run(debug=True, port=8080)
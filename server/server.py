from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import warnings
import pickle
import base64

app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore")

# Load the model

# Create processing function w/ PIL for image

@app.route('/predict', methods=['POST'])
def predict(): 
    data = request.get_json()

    if data is None or 'image' not in data:
        return jsonify({'prediction': 'failed', 'error': 'No image data provided'}), 400

    base64_image = data['image']
    
    if not base64_image.startswith("data:image"):
        return jsonify({'prediction': 'failed', 'error': 'Invalid image data'}), 400

    base64_data = base64_image.split(',')[1]

    image_data = base64.b64decode(base64_data)

    with open('uploaded_image.png', 'wb') as f:
        f.write(image_data)

    return jsonify({'prediction': 'success', 'error': 'null'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)
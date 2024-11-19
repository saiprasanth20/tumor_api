from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'brain_tumor_detector.h5'  # Ensure this file is in the same directory or provide the correct path
model = load_model(MODEL_PATH)

# Preprocessing function for uploaded images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input shape
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])  # Convert BGR to RGB
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file temporarily
    temp_path = 'temp_uploaded_image.jpg'
    file.save(temp_path)

    try:
        # Preprocess the uploaded image
        image = preprocess_image(temp_path)

        # Predict using the model
        prediction = model.predict(image)[0][0]

        # Remove the temporary file
        os.remove(temp_path)

        # Return the prediction result
        result = {
            'prediction': 'Tumor Present' if prediction > 0.5 else 'No Tumor',
            'confidence': float(prediction) if prediction > 0.5 else float(1 - prediction)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a healthcheck endpoint
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'API is running successfully!'})

# Run the Flask app



if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
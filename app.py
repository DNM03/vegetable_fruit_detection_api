from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

model_path = os.getenv('MODEL_PATH', './model/trained_model_upgrade.h5')
cnn = tf.keras.models.load_model(model_path)

test_class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((64, 64))  # Resize to the target size expected by the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert to batch format
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Read and preprocess the image
    image_bytes = file.read()
    input_arr = preprocess_image(image_bytes)
    
    # Make prediction
    predictions = cnn.predict(input_arr)
    predicted_index = np.argmax(predictions[0])
    predicted_class = test_class_names[predicted_index]
    
    # Return the result
    return jsonify({"predicted_class": predicted_class, "confidence": float(predictions[0][predicted_index])})

# Run the app
if __name__ == '__main__':
    app.run()
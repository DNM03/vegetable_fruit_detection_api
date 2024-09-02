from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

model = YOLO(os.getenv('MODEL_PATH', './model/last.pt'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded image temporarily
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)
    
    try:
    # Perform prediction using the YOLO model
        results = model(file_path)

    # Extract class names and probabilities
        names_dict = results[0].names  # Class names
        probs = results[0].probs.data.tolist()  # Probabilities

    # Find the class with the highest probability
        predicted_index = np.argmax(probs)
        predicted_class = names_dict[predicted_index]
        predicted_prob = probs[predicted_index]

    # Return the result as JSON
        return jsonify({
            "predicted_class": predicted_class,
            "predicted_prob": predicted_prob
        }), 200

    except Exception as e:
        return jsonify({"error": e}), 500

    finally:
        # Clean up the saved file after prediction
        if os.path.exists(file_path):
            os.remove(file_path)

# Run the app
if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib  # For loading the pre-trained scaler
import pickle  # For loading the LabelEncoder
from flask_cors import CORS  # For enabling CORS if needed

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS if frontend and backend are on different origins

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="wifi_location_model.tflite")
interpreter.allocate_tensors()

# Load the pre-trained scaler and label encoder
scaler = joblib.load("scaler.pkl")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to process the model and get prediction
def test_model(new_data):
    if len(new_data) != 3:
        raise ValueError("Input data must have exactly 3 signal strength values (Signal_1, Signal_2, Signal_3).")

    # Preprocess the input (scaling)
    new_data_scaled = scaler.transform([new_data])  # Use the pre-trained scaler

    # Prepare the input tensor
    input_tensor = np.array(new_data_scaled, dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)

    # Map the predicted index back to the original label
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

    return predicted_class_label

# Route to serve the index page
@app.route('/')
def index():
    return render_template('index.html')

# API Endpoint to handle POST request with signal data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Ensure we receive the correct signal data
        if 'Signal_1' not in data or 'Signal_2' not in data or 'Signal_3' not in data:
            return jsonify({'error': 'Missing signal data'}), 400

        # Extract the signal data from the request
        signal_data = [data['Signal_1'], data['Signal_2'], data['Signal_3']]

        # Get the prediction from the model
        predicted_class = test_model(signal_data)
        print(f"Predicted Class: {predicted_class}")  # For debugging

        # Return the prediction as a response
        return jsonify({'predicted_class': predicted_class}), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

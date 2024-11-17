from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib
import pickle
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

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

# Store grid data for devices
grid_data = {}  # Format: { "grid_cell_id": ["Device_1", "Device_2"] }

def test_model(new_data):
    if len(new_data) != 3:
        raise ValueError("Input data must have exactly 3 signal strength values (Signal_1, Signal_2, Signal_3).")

    # Preprocess the input (scaling)
    new_data_scaled = scaler.transform([new_data])

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

@app.route('/')
def index():
    return render_template('index.html', grid_data=grid_data)
# Route to serve the index page (for frontend purposes)
@app.route('/Test')
def showPos():
    return render_template('showPos.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'Signal_1' not in data or 'Signal_2' not in data or 'Signal_3' not in data:
            return jsonify({'error': 'Missing signal data'}), 400

        signal_data = [data['Signal_1'], data['Signal_2'], data['Signal_3']]
        predicted_class = test_model(signal_data)

        return jsonify({'predicted_class': predicted_class}), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the request'}), 500
@app.route('/predictShow', methods=['POST'])
def predict_show():
    try:
        data = request.get_json()

        if 'Signal_1' not in data or 'Signal_2' not in data or 'Signal_3' not in data or 'device_name' not in data:
            return jsonify({'error': 'Missing required data'}), 400

        signal_data = [data['Signal_1'], data['Signal_2'], data['Signal_3']]
        device_name = data['device_name']

        predicted_class = test_model(signal_data)

        # Remove the device from its previous position
        for position, devices in grid_data.items():
            if device_name in devices:
                devices.remove(device_name)
                # Clean up the key if no devices remain
                if not devices:
                    del grid_data[position]
                break

        # Add the device to the new position
        if predicted_class not in grid_data:
            grid_data[predicted_class] = []
        if device_name not in grid_data[predicted_class]:
            grid_data[predicted_class].append(device_name)

        return jsonify({'device_name': device_name, 'predicted_class': predicted_class}), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the request'}), 500

@app.route('/getGridData', methods=['GET'])
def get_grid_data():
    return jsonify(grid_data)

if __name__ == '__main__':
    app.run(debug=True)

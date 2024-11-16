import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  # For loading the pre-trained scaler
import pickle  # For loading the LabelEncoder

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="wifi_location_model.tflite")
interpreter.allocate_tensors()

# Load the pre-trained scaler and label encoder
scaler = joblib.load("scaler.pkl")  # Ensure you save the scaler from training as `scaler.pkl`
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to test the model
def test_model(new_data):
    """
    Test the trained TFLite model with new Wi-Fi signal data.
    Args:
        new_data (list or numpy array): List of signal strength values for testing (Signal_1, Signal_2, Signal_3).

    Returns:
        Predicted class label.
    """
    # Validate input dimensions
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

# Example test cases (new data)
test_signals = [
    [-62, -59, -75],  # Replace with actual test data
    [-20, -85, -70],
    [-65, -59, -65],
]

# Run the model on the test cases
print("Testing the model with new signal data:")
for idx, signals in enumerate(test_signals):
    try:
        predicted_class = test_model(signals)
        print(f"Test Case {idx + 1}: Signals: {signals} -> Predicted Class: {predicted_class}")
    except ValueError as e:
        print(f"Test Case {idx + 1}: Error - {str(e)}")

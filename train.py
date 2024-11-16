import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
import pickle

# Load the dataset
df = pd.read_csv('wifi_signal_data_3.csv')

# Remove rows with any -1 in signal columns
df = df[(df['Signal_1'] != -1) & (df['Signal_2'] != -1) & (df['Signal_3'] != -1)]

# Features (signal strengths) and target labels
X = df[['Signal_1', 'Signal_2', 'Signal_3']]
y = df['Additional_Data']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Balance the dataset
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y_encoded)

# Normalize the signal strengths
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Save the fitted scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")

# Save the fitted label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("LabelEncoder saved as 'label_encoder.pkl'")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanced, test_size=0.2, random_state=42)

# Define an improved neural network model
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Early stopping and learning rate scheduler
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, lr_scheduler]
)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Model accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('wifi_location_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite and saved as 'wifi_location_model.tflite'")

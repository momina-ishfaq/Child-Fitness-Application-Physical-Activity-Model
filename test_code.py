import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# Load the test data
data = {
    'Acceleration x (m/s^2)': [0.222000018],
    'Acceleration y (m/s^2)': [5.227050304],
    'Acceleration z (m/s^2)': [9.282000542],
    'Absolute acceleration (m/s^2)': [10.65489901],
    'Gyroscope x (rad/s)': [0],
    'Gyroscope y (rad/s)': [0],
    'Gyroscope z (rad/s)': [0],
    'Absolute (rad/s)': [0],
    'activity_type': ['steps']
}

df = pd.DataFrame(data)

# Convert categorical labels to numerical format
label_encoder = LabelEncoder()


# Encode 'Activity_type' column
df['activity_type'] = label_encoder.fit_transform(df['activity_type'])  # New label 'Activity_type'

# Normalize features
scaler = StandardScaler()
feature_columns = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)',
                   'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)', 'Gyroscope x (rad/s)',
                   'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)', 'Absolute (rad/s)', 'activity_type']

df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Load the trained TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="activity_type_model(our).tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ensure the input data shape matches the model input shape
input_shape = input_details[0]['shape']
if any(input_shape[i] != df[feature_columns].shape[i] for i in range(len(input_shape))):
    raise ValueError(f"Input shape mismatch, expected {input_shape}, but got {df[feature_columns].shape}")

# Prepare input data for inference
input_data = df[feature_columns].values.astype(np.float32)

# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Retrieve the output
output_data = interpreter.get_tensor(output_details[0]['index'])

# Perform post-processing if necessary

print("Output predictions:", output_data)

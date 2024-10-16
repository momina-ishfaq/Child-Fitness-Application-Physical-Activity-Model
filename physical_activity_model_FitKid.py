import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report

# Read the combined dataset
df = pd.read_csv('combined_dataset.csv')

# 1. Data Cleaning
# Check for missing values
print(df.isnull().sum())

# 2. Data Preprocessing
# Convert categorical labels to numerical format
label_encoder = LabelEncoder()
df['activity_type'] = label_encoder.fit_transform(df['activity_type'])  # New label 'activity_type'

# Normalize features
scaler = StandardScaler()
feature_columns = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
                   'Absolute acceleration (m/s^2)', 'Gyroscope x (rad/s)', 'Gyroscope y (rad/s)',
                   'Gyroscope z (rad/s)', 'Absolute (rad/s)', 'BMI']
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Split the dataset into features (X) and labels (y)
X = df[feature_columns]
y = df['activity_type']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training with TensorFlow
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Updated to match the number of activity types
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 4. Model Evaluation
predictions = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, predictions))

# 5. Convert TensorFlow model to TensorFlow Lite (TFLite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('activity_type_model.tflite', 'wb') as f:  # Change the filename
    f.write(tflite_model)

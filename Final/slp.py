import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define paths and class labels
dataset_path = "./dataset"
class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Define symptom mapping
symptom_mapping = {
    "0": "Headache", "1": "Stomach Ache", "2": "Fever", "3": "Nausea",
    "4": "Fatigue", "5": "Pain", "6": "High Temperature", "7": "Wounds",
    "8": "Ears", "9": "Throat"
}

def extract_landmarks_from_image(image_path):
    """Extract hand landmarks from an image."""
    import cv2
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return data
    return None

def load_data():
    """Load dataset and extract landmarks."""
    X, y = [], []
    for label in class_labels:
        class_dir = os.path.join(dataset_path, label)
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            landmarks = extract_landmarks_from_image(img_path)

            if landmarks is not None:
                X.append(landmarks)
                y.append(int(label))

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(class_labels))
    print(f"Loaded {len(X)} samples.")
    return X, y

# Load data and split into training and testing sets
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(class_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, 
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("sign_language_model.h5")
print("Model saved as 'sign_language_model.h5'.")

# Plot Training Results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

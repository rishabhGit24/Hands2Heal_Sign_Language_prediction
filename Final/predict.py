import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model("sign_language_model.h5")
class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Define symptom mapping
symptom_mapping = {
    "0": "Headache",
    "1": "Stomach Ache",
    "2": "Fever",
    "3": "Nausea",
    "4": "Fatigue",
    "5": "Pain",
    "6": "High Temperature",
    "7": "Wounds",
    "8": "Ears",
    "9": "Throat"
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get the input shape from the model
input_shape = model.input_shape[1:]
IMG_SIZE = input_shape[:2]

def preprocess_landmarks(landmarks):
    """Flatten and normalize hand landmarks for model input."""
    data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    data -= np.mean(data)  # Normalize
    data /= np.std(data)   # Scale
    return np.expand_dims(data, axis=0)

def predict_from_hand(frame):
    """Detect hand landmarks, preprocess, and predict."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        input_data = preprocess_landmarks(landmarks)
        
        # Predict using the trained model
        predictions = model.predict(input_data)[0]
        predicted_index = np.argmax(predictions)

        # Print predictions for debugging
        print("Raw predictions:", predictions)
        print("Predicted index:", predicted_index)

        confidence_scores = {class_labels[i]: predictions[i] for i in range(len(class_labels))}
        sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)

        top_prediction = sorted_scores[0]
        predicted_number, confidence = top_prediction
        predicted_symptom = symptom_mapping[predicted_number]

        return f"{predicted_number} - {predicted_symptom} (Confidence: {confidence:.2f})", sorted_scores
    return "No hand detected", []

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        predicted_label, all_scores = predict_from_hand(frame)
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display top 3 predictions
        for i, (label, score) in enumerate(all_scores[:3]):
            text = f"{label} - {symptom_mapping[label]}: {score:.2f}"
            cv2.putText(frame, text, (10, 60 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        cv2.imshow("Sign Language Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

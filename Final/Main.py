import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import google.generativeai as genai
from deep_translator import GoogleTranslator

# Load the trained model and set up MediaPipe Hands
tf_model = tf.keras.models.load_model("sign_language_model.h5")
class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

symptom_mapping = {
    "0": "Headache", "1": "Stomach Ache", "2": "Fever", "3": "Nausea",
    "4": "Fatigue", "5": "Pain", "6": "High Temperature",
    "7": "Wounds", "8": "Ears", "9": "Throat"
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Set up Gemini API
genai.configure(api_key="AIzaSyAK3YtQdH5qNnxY6DSXyJ3luaEydGAUsXA")
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Add the missing functions
def preprocess_landmarks(landmarks):
    # Flatten the landmarks into a 1D array
    flattened = np.array([[l.x, l.y, l.z] for l in landmarks]).flatten()
    # Normalize the coordinates
    normalized = (flattened - np.min(flattened)) / (np.max(flattened) - np.min(flattened))
    return normalized

def predict_from_hand(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Get the landmarks of the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        
        # Preprocess the landmarks
        preprocessed = preprocess_landmarks(hand_landmarks)
        
        # Reshape the input for the model
        input_data = np.expand_dims(preprocessed, axis=0)
        
        # Make a prediction using the TensorFlow model
        prediction = tf_model.predict(input_data)
        predicted_class = np.argmax(prediction)
        
        # Map the predicted class to a symptom
        predicted_symptom = symptom_mapping[class_labels[predicted_class]]
        
        return f"Predicted: {class_labels[predicted_class]} - {predicted_symptom}", prediction[0][predicted_class]
    else:
        return "No hand detected", 0.0

def get_disease_info(symptoms):
    try:
        chat_session = gemini_model.start_chat(history=[])
        prompt = (
            f"Patient has symptoms: {symptoms}. "
            "Provide only the disease name and the possible cure, including medication names, without any disclaimers or advice."
        )
        response = chat_session.send_message(prompt)
        return response.text.replace('*', '').strip()
    except Exception as e:
        return f"Sorry, I couldn't fetch the information right now. Error: {str(e)}"

def translate_response(response, lang_codes=['hi', 'kn']):
    translations = {}
    for lang in lang_codes:
        try:
            translated_text = GoogleTranslator(source='auto', target=lang).translate(response)
            translations[lang] = translated_text
        except Exception as e:
            translations[lang] = f"Translation not available. Error: {str(e)}"
    return translations

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detected_symptoms = []
    print("Press 'q' to quit, 's' to stop detecting and get results.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            predicted_label, _ = predict_from_hand(frame)
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Sign Language Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Stop detection and process results
                break

            # Extract symptom from prediction and add to list
            if "No hand detected" not in predicted_label:
                symptom = predicted_label.split('-')[1].split('(')[0].strip()
                if symptom not in detected_symptoms:
                    detected_symptoms.append(symptom)

        # Process detected symptoms
        if detected_symptoms:
            symptoms_str = ", ".join(detected_symptoms)
            print(f"\nDetected symptoms: {symptoms_str}")
            
            print("\n🤖 Fetching information... please wait a moment.")
            disease_info = get_disease_info(symptoms_str)
            
            if ':' in disease_info:
                disease_name, treatment = disease_info.split(':', 1)
            else:
                disease_name, treatment = disease_info, "No treatment information available."
            
            translations_disease = translate_response(disease_name.strip())
            translations_treatment = translate_response(treatment.strip())
            
            print("\n🌡 Possible Disease and Treatment:")
            print(f"English:")
            print(f"Disease Name: {disease_name.strip()}")
            print(f"Possible Cure: {treatment.strip()}")
            
            print("\nHindi:")
            print(f"रोग का नाम: {translations_disease['hi']}")
            print(f"संभव इलाज: {translations_treatment['hi']}")
            
            print("\nKannada:")
            print(f"ರೋಗದ ಹೆಸರು: {translations_disease['kn']}")
            print(f"ಸಂಭಾವ್ಯ ಚಿಕಿತ್ಸೆ: {translations_treatment['kn']}")
        else:
            print("No symptoms were detected.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

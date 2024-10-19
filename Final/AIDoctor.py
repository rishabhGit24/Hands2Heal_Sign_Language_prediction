import google.generativeai as genai
from deep_translator import GoogleTranslator

# Set up the API key (use your own key here)
genai.configure(api_key="AIzaSyAK3YtQdH5qNnxY6DSXyJ3luaEydGAUsXA")  # Replace with your actual API key

# Create the generation configuration
generation_config = {
    "temperature": 0.7,  # Control randomness
    "top_p": 0.95,
    "top_k": 50,
    "max_output_tokens": 512,
}

# Start a conversation with the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Define the function to get disease info based on symptoms
def get_disease_info(symptoms):
    try:
        # Start a chat session with the model
        chat_session = model.start_chat(history=[])
        prompt = (
            f"Patient has symptoms: {symptoms}. "
            "Provide only the disease name and the possible cure, including medication names, without any disclaimers or advice."
        )

        # Get the model's response
        response = chat_session.send_message(prompt)
        cleaned_response = response.text.replace('*', '').strip()  # Remove asterisks and extra whitespace
        
        return cleaned_response
    except Exception as e:
        return f"Sorry, I couldn't fetch the information right now. Error: {str(e)}"

# Function to translate the response into different languages
def translate_response(response, lang_codes=['hi', 'kn']):
    translations = {}
    for lang in lang_codes:
        try:
            translated_text = GoogleTranslator(source='auto', target=lang).translate(response)
            translations[lang] = translated_text
        except Exception as e:
            translations[lang] = f"Translation not available. Error: {str(e)}"
    return translations

# Friendly chatbot interaction
def chatbot_interaction():
    print("Welcome to the AI Doctor Chatbot! 🏥")
    print("I'm here to help you find possible diseases and treatments based on your symptoms.")
    
    # Get the user's symptoms
    symptoms = input("\nPlease enter your symptoms (e.g., fever, cough, fatigue): ")
    
    # Get disease information and treatment
    print("\n🤖 Fetching information... please wait a moment.")
    disease_info = get_disease_info(symptoms)
    
    # Handle cases where no information is found
    if not disease_info or "no information" in disease_info.lower():
        print("Sorry, I couldn't find any disease matching those symptoms. Please try again.")
        return
    
    # Split the disease information into disease name and treatment
    if ':' in disease_info:
        disease_name, treatment = disease_info.split(':', 1)
    else:
        disease_name, treatment = disease_info, "No treatment information available."
    
    # Translate the disease name and treatment
    translations_disease = translate_response(disease_name.strip())
    translations_treatment = translate_response(treatment.strip())
    
    # Output the results in a conversational format
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
    
    print("\n💡 Stay healthy! Always verify the treatment by consulting a doctor.")

# Run the chatbot interaction
chatbot_interaction()

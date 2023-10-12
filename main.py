import time
import openai

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import time



face_classifier = cv2.CascadeClassifier(r"C:\Users\Maheshi\Desktop\Fer\haarcascade_frontalface_default.xml")
classifier_model = load_model(r"C:\Users\Maheshi\Desktop\Fer\model.h5")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
longterm_emotions = {'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0,
                     'neutral': 0.0}

cap = cv2.VideoCapture(0)
t_end = time.time() + 6

detected_emotions = []
smoothing_window_size = 5  # Adjust this value for the desired level of smoothing

while time.time() < t_end:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier_model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            detected_emotions.append(label)

            # Apply smoothing
            if len(detected_emotions) > smoothing_window_size:
                detected_emotions.pop(0)  # Remove the oldest emotion if the list is too long

            smoothed_emotion = max(set(detected_emotions), key=detected_emotions.count)
            label_position = (x, y)
            cv2.putText(frame, smoothed_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate the final smoothed emotion
if detected_emotions:
    final_smoothed_emotion = max(set(detected_emotions), key=detected_emotions.count)
    print("Smoothed Emotion:", final_smoothed_emotion)

else:
    print("No emotions detected during the given time.")
    exit()

api_key = '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'
openai.api_key = api_key

conversation_history=[]
def get_gpt_response(user_input, emotion):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f'pretend you are an experienced counselor and chat. {user_input} ',
            max_tokens=500,  # Adjust the response length as needed
        )

        return response.choices[0].text
    except Exception as e:
        print("Error:", e)
        return "Sorry, there was an issue with generating a response. Please try again."

conversation_history = []  # Store the conversation history

def continuous_chat(final_smoothed_emotion):
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'stop':
            break  # Exit the loop if the user says "stop"

        # Get the latest detected emotion
        detected_emotion = final_smoothed_emotion

        # Append the user's message to the conversation history
        conversation_history.append(f'You: {user_input}')

        # Get GPT-3.5's response based on the conversation history
        chatbot_response = get_gpt_response(conversation_history, detected_emotion)


        # Append the chatbot's response to the conversation history
        conversation_history.append(f'Chatbot: {chatbot_response}')

        print("Chatbot:", chatbot_response)

if final_smoothed_emotion == 'sad':
    chatbot_question = "I'm sorry to hear that you're feeling sad. Can you talk about what's troubling you?", "I'm here to help. What's on your mind?"
elif final_smoothed_emotion == 'angry':
    chatbot_question = "I'm here to listen. What's making you angry?", "Tell me more about what's bothering you."

elif final_smoothed_emotion == 'disgust':
    chatbot_question = "I'm here to support you. What's causing your feelings of disgust?", "Feel free to share your concerns with me."
elif final_smoothed_emotion == 'fear':
    chatbot_question = "What makes you afraid"
elif final_smoothed_emotion == 'surprise':
    chatbot_question = "What surprised you?"
elif final_smoothed_emotion == 'neutral':
    chatbot_question = 'How can I assist you today?'
elif final_smoothed_emotion == 'happy':
    chatbot_question = 'Hi tell me what makes you happy today?'
else:
    chatbot_question = 'Good bye'
print(chatbot_question)


continuous_chat(final_smoothed_emotion)

from flask import Flask,request,Response,jsonify
from flask_cors import CORS
import openai
import json
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import time

app = Flask(__name__)
CORS(app)

api_key = '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'
openai.api_key = api_key

face_classifier = cv2.CascadeClassifier(r"C:\Users\Maheshi\Desktop\Fer\haarcascade_frontalface_default.xml")
classifier_model = load_model(r"C:\Users\Maheshi\Desktop\Fer\model.h5")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def detect_emotion():
    cap = cv2.VideoCapture(0, cv2.CAP_GPHOTO2)
    smoothing_window_size = 5
    detected_emotions=[]

    t_end = time.time() + 6

    while time.time()<t_end:
        _, frame = cap.read()
        if not _:
            print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            break
        if frame is not None:
            
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray) 



            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
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

            cv2.imshow('Emotion Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('No frame is captured')
            break

    
    cap.release()
    cv2.destroyAllWindows()
    if detected_emotions:
        final_smoothed_emotion = max(set(detected_emotions), key=detected_emotions.count)
    else:
        final_smoothed_emotion = 'neutral'  # Default emotion if none detected

    return final_smoothed_emotion


def get_gpt_response(user_input,detected_emotion,conversation_history):
    try:
        if detected_emotion == 'sad':
            chatbot_question = "I'm sorry to hear that you're feeling sad. Can you talk about what's troubling you?"
        elif detected_emotion == 'angry':
            chatbot_question = "I'm here to listen. What's making you angry?"
        elif detected_emotion == 'disgust':
            chatbot_question = "I'm here to support you. What's causing your feelings of disgust?"
        elif detected_emotion == 'fear':
            chatbot_question = "What makes you afraid?"
        elif detected_emotion == 'surprise':
            chatbot_question = "What surprised you today?"
        elif detected_emotion == 'neutral':
            chatbot_question = 'How can I assist you today?'
        elif detected_emotion == 'happy':
            chatbot_question = 'Hi, tell me what makes you happy today?'
        else:
            chatbot_question = 'Good bye'
        prompt = f'pretend you are an experienced counselor and chat. I am feeling {detected_emotion} because {user_input}\n{conversation_history}\nUser: {user_input}\nChatGPT: {chatbot_question[0]}'
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,  # Adjust the response length as needed
        )
        return response.choices[0].text
    except Exception as e:
        print(e)
        return "Sorry, there was an issue with generating a response. Please try again."

detected_emotion = detect_emotion()
    
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = json.loads(request.data)
        user_input = data['user_input']
        chat_history = data['chat_history']
        #detected_emotion = detect_emotion()
        response = get_gpt_response(user_input,detected_emotion,chat_history)
        return Response(response=response, status=200, mimetype='application/json')
    except Exception as e:
        print(e)
        return Response(response="Sorry, there was an issue with generating a response. Please try again.", status=500, mimetype='application/json')

@app.route('/api/get_initial_question', methods=['GET'])
def get_initial_question():
    
    #detected_emotion = detect_emotion()
    conversation_history = ''
    initial_question = get_gpt_response('',detected_emotion,conversation_history)
    return jsonify({'initial_question': initial_question})
        

if __name__ == '__main__':
    app.run(debug=True)




import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('facial_emotion_model_02.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised'] 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("error, can not open camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("error, can not capture!")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi_gray = gray_frame[y:y+h, x:x+w]
        
        resized_face = cv2.resize(face_roi_gray, (128, 128))
        face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2RGB)
        normalized_face = face_rgb / 255.0
        model_input = np.reshape(normalized_face, (1, 128, 128, 3))

        prediction = model.predict(model_input)     
        max_index = np.argmax(prediction[0])
        predicted_emotion = emotion_labels[max_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
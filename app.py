import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

# Load model once
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("best_emotion_model_80.h5")
    except:
        return tf.keras.models.load_model("emotion_model_improved.h5")

model = load_model()

emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face):
    face_resized = cv2.resize(face, (48,48))
    face_equalized = cv2.equalizeHist(face_resized)
    face_normalized = face_equalized / 255.0
    return np.expand_dims(face_normalized.reshape(48,48,1), axis=0)

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        img = preprocess_face(roi)
        preds = model.predict(img, verbose=0)[0]

        emotion = emotions[np.argmax(preds)]
        conf = np.max(preds) * 100

        # Box + Label
        color = (0,255,0)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{emotion} {conf:.1f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

st.title("😊 Real-Time Emotion Detection App")
st.write("Detect emotions live through your webcam!")

# Webcam input
camera = st.camera_input("Capture a photo")

if camera is not None:
    img = Image.open(camera)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    result = predict_emotion(frame)
    st.image(result, channels="BGR", caption="Detected Faces")

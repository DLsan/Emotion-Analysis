import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

st.title("🎭 Real-Time Emotion Detection")
st.write("Turn on your webcam and let the model detect facial emotions!")

# ✅ Load model once (Streamlit compatible caching)@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_tf")

model = load_model()


emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ✅ Prevent crash if Haarcascade missing
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_cascade()

img = st.camera_input("📸 Capture your face")

if img:
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Failed to process image")
        st.stop()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = face.reshape(1, 48, 48, 1)

        preds = model.predict(face, verbose=0)
        idx = np.argmax(preds)
        emotion = emotions[idx]
        confidence = preds[0][idx] * 100

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    st.image(frame, channels="BGR", caption="Result")

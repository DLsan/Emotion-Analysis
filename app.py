from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

app = Flask(__name__)

# Load the trained model (use the best model from training)
MODEL_PATH = 'best_emotion_model_80.h5'

# Try loading the best model, fall back to regular model if not found
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Loaded model: {MODEL_PATH}")
elif os.path.exists('emotion_model_improved.h5'):
    model = tf.keras.models.load_model('emotion_model_improved.h5')
    MODEL_PATH = 'emotion_model_improved.h5'
    print(f"✓ Loaded model: emotion_model_improved.h5")
elif os.path.exists('emotion_model.h5'):
    model = tf.keras.models.load_model('emotion_model.h5')
    MODEL_PATH = 'emotion_model.h5'
    print(f"✓ Loaded model: emotion_model.h5")
else:
    print("❌ ERROR: No trained model found!")
    print("Please train a model first by running: python download_model.py")
    exit(1)

# Define emotion labels (matching your training script)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion colors for visualization (BGR format)
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 0),    # Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 165, 255), # Orange
    'Neutral': (128, 128, 128) # Gray
}

# Initialize face detection (Haar Cascade)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("❌ ERROR: Could not load face cascade classifier!")
    exit(1)

# Global variable to store emotion statistics
emotion_stats = {emotion: 0 for emotion in emotions}
total_detections = 0

def preprocess_face(face_roi):
    """
    Preprocess face image to match training preprocessing
    This MUST match the preprocessing in download_model.py
    """
    # Resize to 48x48
    face_resized = cv2.resize(face_roi, (48, 48))
    
    # Apply histogram equalization (matching training)
    face_equalized = cv2.equalizeHist(face_resized)
    
    # Normalize to [0, 1]
    face_normalized = face_equalized / 255.0
    
    # Reshape for model input: (1, 48, 48, 1)
    face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))
    
    return face_reshaped

def detect_emotion(frame):
    """
    Detect faces and predict emotions in the frame
    """
    global emotion_stats, total_detections
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # How much image size is reduced at each scale
        minNeighbors=5,     # How many neighbors each rectangle should have
        minSize=(30, 30),   # Minimum face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess face (same as training)
        processed_face = preprocess_face(face_roi)
        
        # Predict emotion
        prediction = model.predict(processed_face, verbose=0)
        
        # Get emotion with highest probability
        emotion_idx = np.argmax(prediction[0])
        emotion = emotions[emotion_idx]
        confidence = prediction[0][emotion_idx] * 100
        
        # Update statistics
        emotion_stats[emotion] += 1
        total_detections += 1
        
        # Get color for this emotion
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Prepare text
        text = f"{emotion}: {confidence:.1f}%"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width, y),
            color,
            -1  # Filled rectangle
        )
        
        # Draw emotion text with white color
        cv2.putText(
            frame,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text
            2
        )
        
        # Show all emotion probabilities (optional - for debugging)
        if False:  # Set to True to see all probabilities
            y_offset = y + h + 20
            for i, (emo, prob) in enumerate(zip(emotions, prediction[0])):
                prob_text = f"{emo}: {prob*100:.1f}%"
                cv2.putText(
                    frame,
                    prob_text,
                    (x, y_offset + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
    
    # Add info overlay
    num_faces = len(faces)
    info_text = f"Faces: {num_faces} | Model: {Path(MODEL_PATH).stem}"
    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    
    return frame

def generate_frames():
    """
    Generator function to stream video frames
    """
    # Try to open webcam (0 is default camera)
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ ERROR: Could not open camera!")
        return
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened successfully")
    
    try:
        while True:
            # Read frame from camera
            success, frame = camera.read()
            
            if not success:
                print("⚠ Failed to read frame from camera")
                break
            
            # Process frame for emotion detection
            processed_frame = detect_emotion(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            
            if not ret:
                print("⚠ Failed to encode frame")
                continue
            
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        # Release camera when done
        camera.release()
        print("✓ Camera released")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', emotions=emotions)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stats')
def get_stats():
    """Get emotion statistics"""
    global emotion_stats, total_detections
    
    # Calculate percentages
    stats_with_percent = {}
    for emotion, count in emotion_stats.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        stats_with_percent[emotion] = {
            'count': count,
            'percentage': round(percentage, 2)
        }
    
    return jsonify({
        'stats': stats_with_percent,
        'total': total_detections
    })

@app.route('/reset_stats')
def reset_stats():
    """Reset emotion statistics"""
    global emotion_stats, total_detections
    emotion_stats = {emotion: 0 for emotion in emotions}
    total_detections = 0
    return jsonify({'status': 'success', 'message': 'Statistics reset'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FER-2013 Emotion Detection - Real-time Web App")
    print("="*70)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Emotions: {', '.join(emotions)}")
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, threaded=True)
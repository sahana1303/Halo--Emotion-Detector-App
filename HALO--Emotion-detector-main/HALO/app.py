from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
from deepface import DeepFace
import threading
import numpy as np
import atexit
import time

app = Flask(__name__)

# Define emotion-based recommendations
emotion_recommendations = {
    "happy": {
        "type": "song",
        "message": "You're glowing with happiness! Enjoy this feel-good song 🎵",
        "link": "https://www.youtube.com/watch?v=y6Sxv-sUYtM",  # Happy - Pharrell Williams
        "cover_image": "https://img.youtube.com/vi/y6Sxv-sUYtM/0.jpg"
    },
    "sad": {
        "type": "video",
        "message": "Feeling low? Here's a motivational video to lift your spirits 💪",
        "link": "https://www.youtube.com/watch?v=0O65HU5BqiA",  # Motivational Video
        "cover_image": "https://img.youtube.com/vi/0O65HU5BqiA/0.jpg"
    },
    "angry": {
        "type": "video",
        "message": "Take a deep breath. Here's a calming video to relax 🧘‍♂️",
        "link": "https://www.youtube.com/watch?v=ZfPISsIIKQw",  # Sandeep Maheshwari Motivational Video
        "cover_image": "https://img.youtube.com/vi/ZfPISsIIKQw/0.jpg"
    },
    "surprise": {
        "type": "book",
        "message": "Surprise! How about exploring this intriguing book 📖",
        "link": "https://www.goodreads.com/book/show/59514.The_Alchemist",
        "cover_image": "https://images.gr-assets.com/books/1403181133l/59514.jpg"
    },
    "fear": {
        "type": "video",
        "message": "Fear not! Watch this for some courage 🌈",
        "link": "https://www.youtube.com/watch?v=MIr3RsUWrdo",  # Guided Meditation
        "cover_image": "https://img.youtube.com/vi/MIr3RsUWrdo/0.jpg"
    },
    "neutral": {
        "type": "video",
        "message": "Here's something to gently stimulate your mind 🌼",
        "link": "https://www.youtube.com/watch?v=J---aiyznGQ",
        "cover_image": "https://img.youtube.com/vi/J---aiyznGQ/0.jpg"
    },
    "disgust": {
        "type": "video",
        "message": "Let’s change that mood. Here's something refreshing 🍃",
        "link": "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
        "cover_image": "https://img.youtube.com/vi/2Vv-BfVoq4g/0.jpg"
    }
}

emotion_result = None
lock = threading.Lock()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()


def release_camera():
    print("Releasing camera...")
    cap.release()


atexit.register(release_camera)

last_frame = None


def detect_emotions():
    global emotion_result
    time.sleep(2)  # wait for camera warm-up
    while True:
        if last_frame is not None:
            try:
                rgb_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(rgb_frame, (640, 480))

                analysis = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)
                if analysis and 'dominant_emotion' in analysis[0]:
                    dominant_emotion = analysis[0]['dominant_emotion']
                    confidence = max(analysis[0]['emotion'].values())
                    with lock:
                        if emotion_result is None:  # Only update if emotion_result is not set (after Clear)
                            emotion_result = dominant_emotion if confidence > 30 else "Uncertain"
                    print("Detected Emotion:", emotion_result)
            except Exception as e:
                print("Emotion Detection Error:", e)
                with lock:
                    emotion_result = "No face detected"
        time.sleep(2)


def generate_frames():
    global last_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        last_frame = frame.copy()

        with lock:
            if emotion_result:
                cv2.putText(frame, f"{emotion_result}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clear', methods=['POST'])
def clear():
    global emotion_result
    with lock:
        emotion_result = None  # Reset emotion_result to allow re-detection
    return jsonify({"message": "Cleared"}), 200


@app.route('/get_emotion')
def get_emotion():
    global emotion_result
    with lock:
        recommendation = emotion_recommendations.get(emotion_result, {
            "type": "video",
            "message": "Here's something interesting!",
            "link": "https://www.youtube.com/",
            "cover_image": ""
        })
        return jsonify({
            "emotion": emotion_result,
            "recommendation": recommendation
        })


@app.route('/recommend', methods=['POST'])
def recommend():
    global emotion_result
    with lock:
        if emotion_result is None:
            return jsonify({"message": "Please detect an emotion first."}), 400

        recommendation = emotion_recommendations.get(emotion_result, {
            "type": "video",
            "message": "Here's something interesting!",
            "link": "https://www.youtube.com/",
            "cover_image": ""
        })
        return jsonify({
            "emotion": emotion_result,
            "recommendation": recommendation
        })


# Login Routes
users = {"admin": "password"}  # Example user credentials


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users.get(username) == password:
            return redirect(url_for('index'))  # Redirect to the main page after successful login
        else:
            return "Invalid Credentials", 403

    return render_template('login.html')


if __name__ == "__main__":
    threading.Thread(target=detect_emotions, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)

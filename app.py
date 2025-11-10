# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from deepface import DeepFace
import io
from PIL import Image
import time

app = Flask(__name__)
last_detection = {"emotion": "neutral", "greeting": "Hello there!", "time": 0}
DETECTION_INTERVAL = 3.0  # seconds

def get_greeting(emotion):
    greetings = {
        "happy": "You look happy today!",
        "sad": "Don't worry, things will get better.",
        "angry": "Take a deep breath. Calm down.",
        "surprise": "Wow! You seem surprised.",
        "neutral": "You look calm and neutral.",
        "fear": "You look a bit scared, everything’s okay!",
        "disgust": "Hmm, something smells bad?"
    }
    return greetings.get(emotion.lower(), "Hello there!")

def base64_to_cv2_img(base64_str):
    # base64_str: data:image/jpeg;base64,/...
    header, encoded = base64_str.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(data)).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global last_detection
    now = time.time()
    # Throttle detection to DETECTION_INTERVAL
    if now - last_detection["time"] < DETECTION_INTERVAL:
        return jsonify(last_detection)

    data = request.json
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({"error": "no image"}), 400

    try:
        frame = base64_to_cv2_img(img_b64)
        # DeepFace returns a dict; enforce_detection False to avoid exceptions if face not found
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # DeepFace.analyze can return a dict for single face or list for multiple; handle both
        if isinstance(result, list):
            dominant = result[0].get('dominant_emotion', 'neutral')
        else:
            dominant = result.get('dominant_emotion', 'neutral')
        greeting = get_greeting(dominant)
        last_detection = {"emotion": dominant, "greeting": greeting, "time": now}
        return jsonify(last_detection)
    except Exception as e:
        # return last detection if analysis fails
        return jsonify({"error": str(e), **last_detection}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


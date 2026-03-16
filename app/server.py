import sys
import os
import base64
import pickle
import time
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from capture import get_frame, draw_landmarks, detector
from landmarks import extract_normalized_landmarks, get_landmarks_from_result
from smooth import PredictionSmoother

app = Flask(__name__, static_folder='static', template_folder='static')
app.config['SECRET_KEY'] = 'signbridge-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'asl_model.pkl')

# ── Load model ──
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f'[SignBridge] Model loaded from {MODEL_PATH}')
except FileNotFoundError:
    print(f'[SignBridge] ERROR: Model not found at {MODEL_PATH}')
    print('[SignBridge] Run python src/train.py first')
    model = None

# ── Globals ──
streaming = False
stream_thread = None
smoother = PredictionSmoother(buffer_size=5, min_confidence=0.75)


def stream_loop():
    """Capture frames, run inference, emit to all connected clients."""
    global streaming
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    frame_count = 0

    print('[SignBridge] Stream started')

    while streaming:
        frame, result = get_frame(cap)
        if frame is None:
            break

        # ── Run inference ──
        prediction_data = None
        lm_raw = get_landmarks_from_result(result)

        if lm_raw and model:
            lm_vector = extract_normalized_landmarks(lm_raw)
            pred = model.predict([lm_vector])[0]
            proba = model.predict_proba([lm_vector])[0]
            confidence = float(proba.max())

            if smoother.is_confident(confidence):
                smoother.update(pred, confidence)

            stable = smoother.get_stable()

            # Build candidates list
            classes = model.classes_
            candidates = sorted(
                [{'letter': str(c), 'confidence': float(p)} for c, p in zip(classes, proba)],
                key=lambda x: x['confidence'],
                reverse=True
            )[:4]

            prediction_data = {
                'letter': stable or pred,
                'confidence': confidence,
                'candidates': candidates
            }

        # ── Draw landmarks ──
        frame = draw_landmarks(frame, result)

        # ── FPS ──
        frame_count += 1
        now = time.time()
        fps = round(1.0 / (now - prev_time)) if (now - prev_time) > 0 else 30
        prev_time = now

        # ── Encode frame ──
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # ── Emit ──
        socketio.emit('frame', {'frame': frame_b64, 'fps': fps})
        if prediction_data:
            socketio.emit('prediction', prediction_data)

        # ~30fps cap
        time.sleep(0.033)

    cap.release()
    print('[SignBridge] Stream stopped')


# ── Routes ──
@app.route('/')
def index():
    return render_template('index.html')


# ── Socket events ──
@socketio.on('connect')
def on_connect():
    global streaming, stream_thread
    print('[SignBridge] Client connected')
    streaming = True
    stream_thread = threading.Thread(target=stream_loop, daemon=True)
    stream_thread.start()
    emit('status', {'message': 'Connected', 'model_loaded': model is not None})


@socketio.on('disconnect')
def on_disconnect():
    global streaming
    print('[SignBridge] Client disconnected')
    streaming = False
    smoother.clear()


@socketio.on('commit_letter')
def on_commit(data):
    """Client explicitly committed a letter."""
    print(f'[SignBridge] Committed: {data.get("letter")}')


if __name__ == '__main__':
    print('[SignBridge] Starting server at http://localhost:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
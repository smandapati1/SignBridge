import base64
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from capture import CONNECTIONS, detect_frame  # noqa: E402
from landmarks import extract_normalized_landmarks, get_landmarks_from_result  # noqa: E402

app = Flask(__name__, static_folder='static', template_folder='static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-change-me')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

MODEL_PATH = ROOT / 'models' / 'asl_model.pkl'
with MODEL_PATH.open('rb') as f:
    model = pickle.load(f)

@app.get('/')
def index():
    return render_template('index.html')

@app.get('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': int(len(model.classes_)),
    })

@app.post('/api/predict')
def predict():
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image', '')
    if not image_data:
        return jsonify({'error': 'Missing image'}), 400

    try:
        encoded = image_data.split(',', 1)[-1]
        raw = base64.b64decode(encoded)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'error': 'Invalid image payload'}), 400

    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400

    result = detect_frame(frame)
    lm_raw = get_landmarks_from_result(result)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    if not lm_raw:
        return jsonify({
            'hand_detected': False,
            'latency_ms': elapsed_ms,
            'connections': CONNECTIONS,
        })

    vector = extract_normalized_landmarks(lm_raw)
    pred = str(model.predict([vector])[0])
    probabilities = model.predict_proba([vector])[0]
    classes = model.classes_
    candidates = sorted(
        ({'letter': str(c), 'confidence': float(p)} for c, p in zip(classes, probabilities)),
        key=lambda item: item['confidence'],
        reverse=True,
    )[:4]

    points = [{'x': float(lm.x), 'y': float(lm.y)} for lm in lm_raw]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return jsonify({
        'hand_detected': True,
        'letter': pred,
        'confidence': float(max(probabilities)),
        'candidates': candidates,
        'landmarks': points,
        'connections': CONNECTIONS,
        'latency_ms': elapsed_ms,
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False)

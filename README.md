# SignBridge

Real-time ASL fingerspelling recognition in the browser using MediaPipe hand landmarks and a Random Forest classifier.

## What makes this version deployable

The browser now owns the webcam through `getUserMedia()`. Frames are sampled and sent to the Flask inference API, which returns only the predicted letter, confidence scores, and landmark coordinates. This avoids the original deployment bug where `cv2.VideoCapture(0)` attempted to open a webcam attached to the cloud server.

### Features

- Browser webcam capture with explicit camera permission
- 26-letter A-Z Random Forest classifier over 63 normalized landmark features
- Live MediaPipe landmark overlay
- Top-4 candidate probabilities
- Optional stable-letter auto-commit
- Manual commit, delete, clear, and spacing controls
- Browser text-to-speech
- Guided fingerspelling practice mode
- Health endpoint for deployment monitoring
- Render Blueprint included

## Architecture

```text
Browser webcam
    ↓
getUserMedia + sampled JPEG frame
    ↓
POST /api/predict
    ↓
MediaPipe Hand Landmarker
21 landmarks × (x, y, z)
    ↓
Wrist-relative + scale normalization
63-dimensional feature vector
    ↓
Random Forest
    ↓
Prediction + confidence + landmarks
    ↓
Browser overlay / auto-commit / TTS
```

## Run locally

Use Python 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_assets.py
python app/server.py
```

Open `http://localhost:5000` and click **Start camera**.

## Deploy to Render

This repository includes `render.yaml`, so the simplest route is a Render Blueprint.

1. Push this version to GitHub.
2. In Render, create a new Blueprint and connect the SignBridge repository.
3. Render will run:

```bash
pip install -r requirements.txt && python scripts/download_assets.py
```

4. It starts the service with:

```bash
gunicorn app.server:app --workers 1 --threads 4 --timeout 120
```

5. When the deploy is healthy, open the public Render URL and allow camera access.

The app exposes `/healthz` for health checks.

## Project structure

```text
SignBridge/
├── app/
│   ├── server.py
│   └── static/
│       └── index.html
├── models/
│   ├── asl_model.pkl
│   └── asl_lstm.pt
├── scripts/
│   └── download_assets.py
├── src/
│   ├── capture.py
│   ├── landmarks.py
│   ├── predict.py
│   ├── smooth.py
│   └── train.py
├── tests/
├── .python-version
├── render.yaml
└── requirements.txt
```

## Model notes

The current model classifies all 26 letters from a single landmark snapshot. This is a strong baseline for static fingerspelling, but **J and Z are motion-dependent signs**, so a future version should use temporal sequences for those letters rather than treating them as purely static poses.

The existing reported 100% holdout result should also be treated as a development metric until evaluation is repeated with signer-independent splits. A better benchmark should report macro F1, per-letter confusion, calibration, and performance on previously unseen signers.

## Highest-value next upgrades

- signer-independent evaluation and confusion matrix
- temporal J/Z classifier using landmark trajectories
- word boundary detection and language-model correction
- browser-side MediaPipe/inference to keep frames fully on-device
- mobile layout and PWA support
- accessibility testing and user feedback from ASL signers

## Tech stack

Python, Flask, Gunicorn, MediaPipe, OpenCV, NumPy, scikit-learn, HTML/CSS/JavaScript, Web Speech API.

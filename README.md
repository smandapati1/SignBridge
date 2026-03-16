# SignBridge

![Python](https://img.shields.io/badge/Python-3.13-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.32-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Accuracy](https://img.shields.io/badge/ASL_Accuracy-100%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Active_Development-blue)

> Real-time American Sign Language recognition via computer vision.  
> Type a word — watch it signed with real ASL dataset images.  
> Point your webcam — get live letter predictions at 30fps.

*This project is actively in development. Core ASL recognition pipeline is complete and functional. Web frontend, LSTM dynamic sign model, and additional modules are in progress.*

---

## What works right now

- **26-letter ASL fingerspelling** — real-time recognition via webcam at 30fps
- **100% test accuracy** — Random Forest classifier on 6,500+ normalized landmark samples
- **Flask web app** — live camera feed streamed over WebSocket to a browser UI
- **Dataset mode** — type any word, watch it signed letter by letter using real ASL dataset images
- **Text-to-speech** — signed output spoken aloud via browser Web Speech API
- **ASL reference chart** — schematic hand diagrams for all 26 letters
- **Confidence scoring** — top 4 candidate predictions with live probability bars

---

## How it works
```
Webcam / Dataset Image
        ↓
MediaPipe HandLandmarker
  21 landmarks × (x, y, z)
        ↓
Coordinate Normalization
  wrist → origin, scale → 1.0
  output: 63-dimensional vector
        ↓
Random Forest Classifier
  n_estimators=100
  trained on 6,500+ samples
        ↓
Prediction Smoothing
  5-frame majority vote
  confidence threshold: 0.75
        ↓
Browser UI via WebSocket
  live feed + prediction overlay
```

---

## Why landmark-based over CNN

Most ASL recognition approaches train directly on raw video frames. SignBridge extracts structured 3D landmarks instead — making the model lighting invariant, position invariant, and fast enough to run on CPU with no GPU required.

---

## Getting started

### Prerequisites
- Python 3.10–3.13
- Webcam
- Windows / Mac / Linux

### Setup
```bash
git clone https://github.com/yourusername/signbridge.git
cd signbridge
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Download MediaPipe model
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### Collect training data
```bash
python asl/src/collect.py
```

### Train
```bash
python asl/src/train.py
```

### Run
```bash
python app/server.py
# open http://localhost:5000
```

---

## Project structure
```
SignBridge/
├── src/
│   ├── capture.py        # MediaPipe pipeline + webcam
│   ├── landmarks.py      # coordinate extraction + normalization
│   ├── collect.py        # data collection tool
│   ├── train.py          # model training + evaluation
│   ├── predict.py        # live inference loop
│   └── smooth.py         # prediction stabilization
├── app/
│   ├── server.py         # Flask + WebSocket backend
│   └── static/
│       └── index.html    # browser frontend
├── data/
│   ├── raw/              # collected landmark CSVs
│   ├── processed/        # merged training data
│   ├── models/           # trained model files
│   └── asl_dataset/      # Kaggle ASL alphabet dataset
├── notebooks/
├── hand_landmarker.task
└── requirements.txt
```

---

## Roadmap

### In progress
- [ ] LSTM model for dynamic signs — proper J, Z, and full word recognition
- [ ] Complete dataset mode for all 26 letters A–Z
- [ ] Word autocomplete from signed letter sequences

### Planned
- [ ] Text-to-speech integration via pyttsx3 (offline)
- [ ] Two-hand support for full ASL grammar
- [ ] ReliefLink module — low-bandwidth disaster relief communication
- [ ] HazardWatch module — air quality and flood prediction via public sensor ML
- [ ] Mobile deployment via TensorFlow Lite

---

## Performance

| Metric | Value |
|---|---|
| Training samples | 6,539 |
| Signs | 26 (A–Z) |
| Test accuracy | 100% |
| Inference speed | ~30fps |
| Model size | <5MB |
| Hardware | CPU only, standard webcam |

---

## Tech stack

| Component | Technology |
|---|---|
| Hand tracking | MediaPipe HandLandmarker |
| Video capture | OpenCV |
| ML model | scikit-learn RandomForestClassifier |
| Data processing | NumPy, pandas |
| Web backend | Flask + Flask-SocketIO |
| Frontend | HTML / CSS / JS |
| TTS | Web Speech API |

---

## Contributing

Contributions welcome — especially ASL dataset samples from diverse signers, additional sign language support, and model improvements.
```bash
git checkout -b feature/your-feature
git commit -m 'add your feature'
git push origin feature/your-feature
# open a pull request
```

---

## License

MIT

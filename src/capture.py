import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    return frame, result

def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return frame
    h, w = frame.shape[:2]
    for hand_landmarks in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for i, pt in enumerate(pts):
            color = (0, 255, 150) if i in [4,8,12,16,20] else (255, 255, 255)
            cv2.circle(frame, pt, 4, color, -1)
    return frame

def run_preview():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        frame, result = get_frame(cap)
        if frame is None:
            break
        frame = draw_landmarks(frame, result)
        cv2.imshow('ASL Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
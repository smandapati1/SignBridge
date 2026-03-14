import cv2

import mediapipe as mp

mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode = False,

    max_num_hands = 1,

    min_detection_confidence = 0.7,

    min_tracking_confidence = 0.5
)

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    return frame, result

def draw_landmarks(frame, result):
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
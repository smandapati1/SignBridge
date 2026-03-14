import numpy as np

def extract_normalized_landmarks(hand_landmarks):
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = pts[0]
    pts = [(p[0]-wrist[0], p[1]-wrist[1], p[2]-wrist[2]) for p in pts]
    max_dist = max(np.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in pts)
    pts = [(p[0]/max_dist, p[1]/max_dist, p[2]/max_dist) for p in pts]
    return np.array(pts).flatten()

def get_landmarks_from_result(result):
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0]
    return None
import cv2
from capture import get_frame, draw_landmarks
from landmarks import extract_normalized_landmarks, get_landmarks_from_result

cap = cv2.VideoCapture(0)

while cap.isOpened():
    frame, result = get_frame(cap)
    if frame is None:
        break

    lm_raw = get_landmarks_from_result(result)
    if lm_raw:
        lm_vector = extract_normalized_landmarks(lm_raw)
        # pass lm_vector to your model here

    frame = draw_landmarks(frame, result)
    cv2.imshow('ASL Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
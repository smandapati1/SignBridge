import csv
import os
import cv2
from capture import hands
from landmarks import extract_normalized_landmarks

SIGNS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DATA_FILE = 'data/raw/asl_data.csv'

def collect_data():
    cap = cv2.VideoCapture(0)
    current_sign = 'A'
    collecting = False

    os.makedirs('data/raw', exist_ok=True)

    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            cv2.putText(frame, f'Sign: {current_sign} | S=record  N=next  Q=quit',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if result.multi_hand_landmarks and collecting:
                lm = extract_normalized_landmarks(result.multi_hand_landmarks[0])
                writer.writerow([current_sign] + lm.tolist())
                cv2.putText(frame, 'RECORDING',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow('Collector', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                collecting = not collecting
            elif key == ord('n'):
                idx = SIGNS.index(current_sign)
                current_sign = SIGNS[(idx + 1) % len(SIGNS)]
                collecting = False
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    collect_data()
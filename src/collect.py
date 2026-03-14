import csv
import os
import cv2
from capture import get_frame, draw_landmarks
from landmarks import extract_normalized_landmarks, get_landmarks_from_result

SIGNS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DATA_FILE = 'data/raw/asl_data.csv'

def collect_data():
    cap = cv2.VideoCapture(0)
    current_sign = 'A'
    collecting = False
    sample_count = 0

    os.makedirs('data/raw', exist_ok=True)

    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)

        while cap.isOpened():
            frame, result = get_frame(cap)
            if frame is None:
                break

            lm_raw = get_landmarks_from_result(result)

            if lm_raw and collecting:
                lm = extract_normalized_landmarks(lm_raw)
                writer.writerow([current_sign] + lm.tolist())
                sample_count += 1
                cv2.putText(frame, f'RECORDING — {sample_count} samples',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            cv2.putText(frame, f'Sign: {current_sign} | S=record  N=next  Q=quit',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame = draw_landmarks(frame, result)
            cv2.imshow('Collector', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                collecting = not collecting
                if not collecting:
                    print(f'Stopped — {sample_count} samples recorded for {current_sign}')
            elif key == ord('n'):
                idx = SIGNS.index(current_sign)
                current_sign = SIGNS[(idx + 1) % len(SIGNS)]
                collecting = False
                sample_count = 0
                print(f'Moved to sign: {current_sign}')
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print('Collection complete.')


if __name__ == '__main__':
    collect_data()
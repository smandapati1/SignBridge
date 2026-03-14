import cv2
import pickle
from capture import get_frame, draw_landmarks, hands
from landmarks import extract_normalized_landmarks, get_landmarks_from_result
from smooth import PredictionSmoother

MODEL_PATH = 'data/models/asl_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def run_prediction():
    model = load_model()
    smoother = PredictionSmoother(buffer_size=5, min_confidence=0.75)
    cap = cv2.VideoCapture(0)
    output_text = ''

    print('Running — press Q to quit, SPACE to add letter to output')

    while cap.isOpened():
        frame, result = get_frame(cap)
        if frame is None:
            break

        stable_pred = None

        lm_raw = get_landmarks_from_result(result)
        if lm_raw:
            lm_vector = extract_normalized_landmarks(lm_raw)
            prediction = model.predict([lm_vector])[0]
            confidence = model.predict_proba([lm_vector])[0].max()

            if smoother.is_confident(confidence):
                smoother.update(prediction, confidence)
                stable_pred = smoother.get_stable()

            # Display current prediction + confidence
            if stable_pred:
                cv2.putText(frame, f'{stable_pred} ({confidence:.0%})',
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            2.5, (0, 255, 150), 3)

        # Display running output text at bottom
        cv2.putText(frame, f'Output: {output_text}',
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        frame = draw_landmarks(frame, result)
        cv2.imshow('SignBridge', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and stable_pred:
            # Press space to commit current letter to output
            output_text += stable_pred
            smoother.clear()
        elif key == ord('z') and output_text:
            # Press z to delete last letter
            output_text = output_text[:-1]

    cap.release()
    cv2.destroyAllWindows()
    print(f'Final output: {output_text}')


if __name__ == '__main__':
    run_prediction()
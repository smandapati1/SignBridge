from collections import Counter

class PredictionSmoother:
    def __init__(self, buffer_size=5, min_confidence=0.7):
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        self.buffer = []

    def update(self, prediction, confidence):
        """Add a new prediction to the buffer."""
        self.buffer.append(prediction)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get_stable(self):
        """Return the most common prediction in the buffer."""
        if not self.buffer:
            return None
        return Counter(self.buffer).most_common(1)[0][0]

    def is_confident(self, confidence):
        """Only accept predictions above confidence threshold."""
        return confidence >= self.min_confidence

    def clear(self):
        self.buffer = []
from pathlib import Path

import numpy as np
import onnxruntime as ort


class Predictor:
    def __init__(self, config):
        self.model_path = Path(config.package.path)
        self.session = ort.InferenceSession(str(self.model_path))
        self.columns = [item.name for item in self.session.get_inputs()]

    def predict(self, record: dict) -> dict:
        missing = [column for column in self.columns if column not in record]
        if missing:
            raise ValueError(f"missing features: {missing}")
        feeds = {
            column: np.array([[float(record[column])]], dtype=np.float32)
            for column in self.columns
        }
        labels, probabilities = self.session.run(None, feeds)
        prediction = int(labels[0])
        scores = probabilities[0]
        return {
            "prediction": prediction,
            "probability": float(scores.get(1, scores[prediction])),
        }

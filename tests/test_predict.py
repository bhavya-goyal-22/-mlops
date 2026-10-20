from mlops.config import load_config
from mlops.predict import Predictor


def test_predictor_scores_onnx(sample):
    result = Predictor(load_config()).predict(sample)

    assert result["prediction"] in {0, 1}
    assert 0 <= result["probability"] <= 1

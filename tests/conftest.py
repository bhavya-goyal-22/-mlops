import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlops.config import load_config


@pytest.fixture
def sample():
    return {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }


@pytest.fixture
def preprocess_config():
    return OmegaConf.create(
        {
            "drop_missing": ["ca", "thal"],
            "target": "target",
            "categorical": ["cp", "restecg", "slope", "thal"],
        }
    )


@pytest.fixture
def dirty_df():
    return pd.DataFrame(
        {
            "age": [63, 67, 41],
            "sex": [1, 1, 0],
            "cp": [1, 4, 3],
            "trestbps": [145, 160, 130],
            "chol": [233, 286, 204],
            "fbs": [1, 0, 0],
            "restecg": [2, 2, 0],
            "thalach": [150, 108, 172],
            "exang": [0, 1, 0],
            "oldpeak": [2.3, 1.5, 1.4],
            "slope": [3, 2, 1],
            "ca": [0.0, None, 0.0],
            "thal": [6.0, 3.0, None],
            "target": [0, 2, 1],
        }
    )


@pytest.fixture
def clean_df():
    rows = []
    for index in range(8):
        rows.append(
            {
                "age": 50 + index,
                "sex": index % 2,
                "cp": (index % 4) + 1,
                "trestbps": 120 + index,
                "chol": 200 + index * 3,
                "fbs": index % 2,
                "restecg": index % 3,
                "thalach": 140 + index,
                "exang": index % 2,
                "oldpeak": index * 0.2,
                "slope": (index % 3) + 1,
                "ca": index % 4,
                "thal": [3, 6, 7, 3][index % 4],
                "target": 0 if index < 4 else 1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.tracking.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg.package.path = str(tmp_path / "model.onnx")
    return cfg

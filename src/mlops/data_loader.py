from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


class DataLoader:

    def __init__(self, config):
        self.dataset_id = config.dataset_id
        self.path = Path(config.path)

    def download(self) -> Path:
        if self.path.exists():
            return self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        dataset = fetch_ucirepo(id=self.dataset_id)
        frame = dataset.data.features.copy()
        frame["target"] = dataset.data.targets
        frame.to_csv(self.path, index=False)
        return self.path

    def load(self) -> pd.DataFrame:
        self.download()
        return pd.read_csv(self.path)
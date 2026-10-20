import pandas as pd


class Preprocessor:
    def __init__(self, config):
        self.drop_missing = list(config.drop_missing)
        self.target = config.target
        self.categorical = list(config.categorical)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.dropna(subset=self.drop_missing).copy()
        for column in self.drop_missing:
            df[column] = df[column].astype(int)
        df[self.target] = (df[self.target] > 0).astype(int)
        return df.reset_index(drop=True)

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(data, columns=self.categorical, drop_first=True)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.encode(self.clean(data))

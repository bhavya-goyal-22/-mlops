from pathlib import Path

from mlops.config import load_config
from mlops.data_loader import DataLoader
from mlops.eda import EDA
from mlops.preprocessing import Preprocessor
from mlops.train import ModelTrainer


def main() -> None:
    config = load_config()
    data = DataLoader(config.data).load()
    clean = Preprocessor(config.preprocess).clean(data)
    cleaned_path = Path(config.data.cleaned_path)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(cleaned_path, index=False)
    EDA(config.eda).run(clean)
    results = ModelTrainer(config).train(clean)
    for name, result in results.items():
        print(name, result["best_params"], result["cv"])


if __name__ == "__main__":
    main()

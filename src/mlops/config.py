from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> DictConfig:
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.data.path = str((ROOT / cfg.data.path).resolve())
    cfg.data.cleaned_path = str((ROOT / cfg.data.cleaned_path).resolve())
    cfg.data.train_path = str((ROOT / cfg.data.train_path).resolve())
    cfg.data.test_path = str((ROOT / cfg.data.test_path).resolve())
    cfg.data.overview_path = str((ROOT / cfg.data.overview_path).resolve())
    cfg.eda.figures_dir = str((ROOT / cfg.eda.figures_dir).resolve())
    cfg.tracking.tracking_uri = f"sqlite:///{(ROOT / cfg.tracking.tracking_uri).resolve()}"
    cfg.package.path = str((ROOT / cfg.package.path).resolve())
    return cfg
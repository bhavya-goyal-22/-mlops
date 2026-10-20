from omegaconf import OmegaConf

from mlops.data_loader import DataLoader
from mlops.preprocessing import Preprocessor


def test_clean_drops_missing_and_binarizes_target(preprocess_config, dirty_df):
    clean = Preprocessor(preprocess_config).clean(dirty_df)

    assert len(clean) == 1
    assert clean["target"].isin([0, 1]).all()
    assert clean.loc[0, "target"] == 0
    assert clean["ca"].dtype == int
    assert clean["thal"].dtype == int


def test_encode_adds_dummy_columns(preprocess_config, clean_df):
    preprocessor = Preprocessor(preprocess_config)
    encoded = preprocessor.encode(clean_df)

    assert "cp" not in encoded.columns
    assert any(column.startswith("cp_") for column in encoded.columns)
    assert "target" in encoded.columns


def test_load_reads_existing_csv(tmp_path, dirty_df):
    path = tmp_path / "heart.csv"
    dirty_df.to_csv(path, index=False)
    loaded = DataLoader(OmegaConf.create({"dataset_id": 45, "path": str(path)})).load()

    assert len(loaded) == 3
    assert "target" in loaded.columns

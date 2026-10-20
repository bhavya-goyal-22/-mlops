import pytest
from sklearn.pipeline import Pipeline

from mlops.train import ModelTrainer


def test_candidates_include_both_models(config):
    names = set(ModelTrainer(config).candidates().keys())
    assert names == {"logistic_regression", "random_forest"}


def test_feature_pipeline_has_expected_transformers(config):
    features = ModelTrainer(config).build_features()
    transformer_names = {name for name, _, _ in features.transformers}
    assert transformer_names == {"numeric", "categorical", "binary"}


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_each_model_fits_and_predicts(config, clean_df, name):
    trainer = ModelTrainer(config)
    estimator, _ = trainer.candidates()[name]
    pipe = Pipeline([("features", trainer.build_features()), ("model", estimator)])
    features = clean_df.drop(columns=["target"])
    labels = clean_df["target"]

    pipe.fit(features, labels)
    predictions = pipe.predict(features)

    assert len(predictions) == len(labels)
    assert set(predictions).issubset({0, 1})

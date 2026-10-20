from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops.config import ROOT
from mlops.eda import EDA


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.cfg = config.model
        self.package_path = Path(config.package.path)
        mlflow.set_tracking_uri(config.tracking.tracking_uri)
        mlflow.set_experiment(config.tracking.experiment)
        mlflow.enable_system_metrics_logging()
        mlflow.sklearn.autolog(log_models=False, log_datasets=True)

    def build_features(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), list(self.cfg.numeric)),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                    list(self.cfg.categorical),
                ),
                ("binary", "passthrough", list(self.cfg.binary)),
            ]
        )

    def candidates(self) -> dict:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=1000, random_state=self.cfg.random_state),
                {"model__C": list(self.cfg.models.logistic_regression.C)},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=self.cfg.random_state),
                {
                    "model__n_estimators": list(self.cfg.models.random_forest.n_estimators),
                    "model__max_depth": list(self.cfg.models.random_forest.max_depth),
                },
            ),
        }

    def save_splits(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        for path, frame in (
            (self.config.data.train_path, train),
            (self.config.data.test_path, test),
        ):
            split_path = Path(path)
            split_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(split_path, index=False)

        overview = pd.concat(
            {
                "all": pd.concat([train, test]).describe(include="all"),
                "train": train.describe(include="all"),
                "test": test.describe(include="all"),
            },
            axis=1,
        )
        overview_path = Path(self.config.data.overview_path)
        overview.to_csv(overview_path)

    def log_shared_artifacts(self) -> None:
        for path, folder in (
            (self.config.data.path, "data"),
            (self.config.data.cleaned_path, "data"),
            (self.config.data.train_path, "data"),
            (self.config.data.test_path, "data"),
            (self.config.data.overview_path, "overview"),
            (ROOT / "configs" / "config.yaml", "config"),
            (ROOT / "requirements.txt", "config"),
        ):
            artifact = Path(path)
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path=folder)
        mlflow.log_artifacts(self.config.eda.figures_dir, artifact_path="eda")

    def log_datasets(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        train_dataset = mlflow.data.from_pandas(
            train,
            source=self.config.data.train_path,
            name="heart_disease_train",
            targets=self.cfg.target,
        )
        test_dataset = mlflow.data.from_pandas(
            test,
            source=self.config.data.test_path,
            name="heart_disease_test",
            targets=self.cfg.target,
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="evaluation")

    def class_distribution_metrics(self, frame: pd.DataFrame, prefix: str) -> dict:
        counts = frame[self.cfg.target].value_counts().sort_index()
        total = max(len(frame), 1)
        return {
            f"{prefix}_rows": float(len(frame)),
            f"{prefix}_class_0_count": float(counts.get(0, 0)),
            f"{prefix}_class_1_count": float(counts.get(1, 0)),
            f"{prefix}_class_0_pct": 100 * float(counts.get(0, 0)) / total,
            f"{prefix}_class_1_pct": 100 * float(counts.get(1, 0)) / total,
        }

    def log_overview(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        train_metrics = self.class_distribution_metrics(train, "train")
        val_metrics = self.class_distribution_metrics(val, "val")
        note = (
            "# Heart disease run overview\n\n"
            f"- Train rows: {int(train_metrics['train_rows'])} | Val rows: {int(val_metrics['val_rows'])} | Val size: {self.cfg.test_size}\n"
            f"- Train class: no disease={int(train_metrics['train_class_0_count'])} ({train_metrics['train_class_0_pct']:.1f}%), "
            f"disease={int(train_metrics['train_class_1_count'])} ({train_metrics['train_class_1_pct']:.1f}%)\n"
            f"- Val class: no disease={int(val_metrics['val_class_0_count'])} ({val_metrics['val_class_0_pct']:.1f}%), "
            f"disease={int(val_metrics['val_class_1_count'])} ({val_metrics['val_class_1_pct']:.1f}%)\n"
            f"- Numeric features: {', '.join(self.cfg.numeric)}\n"
            f"- Categorical features: {', '.join(self.cfg.categorical)}\n"
            f"- Models: logistic regression, random forest\n"
            "- Curves: ROC, precision-recall, and confusion matrix for train and val\n"
        )
        mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_param("train_rows", int(train_metrics["train_rows"]))
        mlflow.log_param("val_rows", int(val_metrics["val_rows"]))
        mlflow.log_param("val_size", self.cfg.test_size)
        mlflow.log_metrics({**train_metrics, **val_metrics})

    def log_curves(self, model, train: pd.DataFrame, val: pd.DataFrame) -> None:
        for split_name, frame in (("train", train), ("val", val)):
            features = frame.drop(columns=[self.cfg.target])
            labels = frame[self.cfg.target]
            predicted = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]

            roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_predictions(labels, probabilities, ax=roc_ax, name=split_name)
            roc_ax.set_title(f"{split_name} ROC curve")
            mlflow.log_figure(roc_fig, f"curves/{split_name}_roc_curve.png")
            plt.close(roc_fig)

            pr_fig, pr_ax = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_predictions(labels, probabilities, ax=pr_ax, name=split_name)
            pr_ax.set_title(f"{split_name} precision-recall curve")
            mlflow.log_figure(pr_fig, f"curves/{split_name}_precision_recall_curve.png")
            plt.close(pr_fig)

            cm_fig, cm_ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay.from_predictions(labels, predicted, ax=cm_ax, colorbar=False)
            cm_ax.set_title(f"{split_name} confusion matrix")
            mlflow.log_figure(cm_fig, f"curves/{split_name}_confusion_matrix.png")
            plt.close(cm_fig)

            mlflow.log_metrics(
                {
                    f"{split_name}_accuracy": float(accuracy_score(labels, predicted)),
                    f"{split_name}_precision": float(precision_score(labels, predicted, zero_division=0)),
                    f"{split_name}_recall": float(recall_score(labels, predicted, zero_division=0)),
                    f"{split_name}_roc_auc": float(roc_auc_score(labels, probabilities)),
                    f"{split_name}_average_precision": float(average_precision_score(labels, probabilities)),
                }
            )

    def train(self, data: pd.DataFrame) -> dict:
        train, test = train_test_split(
            data,
            test_size=self.cfg.test_size,
            stratify=data[self.cfg.target],
            random_state=self.cfg.random_state,
        )
        self.save_splits(train, test)
        EDA(self.config.eda).run_split(train, test, self.cfg.target)

        X_train = train.drop(columns=[self.cfg.target])
        y_train = train[self.cfg.target]
        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds,
            shuffle=True,
            random_state=self.cfg.random_state,
        )
        results = {}

        with mlflow.start_run(run_name="compare-models"):
            self.log_overview(train, test)
            self.log_datasets(train, test)
            self.log_shared_artifacts()

            for name, (estimator, grid) in self.candidates().items():
                pipeline = Pipeline(
                    [
                        ("features", self.build_features()),
                        ("model", estimator),
                    ]
                )
                with mlflow.start_run(run_name=name, nested=True):
                    search = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
                    search.fit(X_train, y_train)
                    mlflow.sklearn.autolog(disable=True)
                    scores = cross_validate(
                        search.best_estimator_,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=list(self.cfg.scoring),
                    )
                    metrics = {
                        f"cv_{metric}": float(scores[f"test_{metric}"].mean())
                        for metric in self.cfg.scoring
                    }
                    model_info = mlflow.sklearn.log_model(
                        search.best_estimator_,
                        name=f"{name}_pipeline",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                    )
                    mlflow.models.evaluate(
                        model=model_info.model_uri,
                        data=test,
                        targets=self.cfg.target,
                        model_type="classifier",
                    )
                    self.log_curves(search.best_estimator_, train, test)
                    mlflow.sklearn.autolog(log_models=False, log_datasets=True)
                    mlflow.log_param("model", name)
                    mlflow.log_params(search.best_params_)
                    mlflow.log_metrics(metrics)
                    self.log_overview(train, test)
                    self.log_datasets(train, test)
                    self.log_shared_artifacts()

                results[name] = {
                    "best_params": search.best_params_,
                    "cv": metrics,
                    "estimator": search.best_estimator_,
                }

            best_name = max(results, key=lambda name: results[name]["cv"]["cv_roc_auc"])
            onnx_path = self.export_onnx(results[best_name]["estimator"], X_train)
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

        return results

    def export_onnx(self, pipeline: Pipeline, X: pd.DataFrame) -> Path:
        self.package_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=[(column, FloatTensorType([None, 1])) for column in X.columns],
        )
        self.package_path.write_bytes(onnx_model.SerializeToString())
        return self.package_path

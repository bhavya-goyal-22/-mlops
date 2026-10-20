from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    def __init__(self, config):
        self.figures_dir = Path(config.figures_dir)
        self.numeric = list(config.numeric)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data) -> None:
        self.class_balance(data)
        self.histograms(data)
        self.correlation_heatmap(data)

    def run_split(self, train, test, target: str) -> None:
        self.train_test_class_balance(train, test, target)
        self.train_test_distributions(train, test)

    def class_balance(self, data) -> None:
        counts = data["target"].value_counts().sort_index()
        labels = ["No disease", "Disease"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, counts.values, color=["#4C78A8", "#E45756"])
        for bar, count in zip(bars, counts.values):
            pct = 100 * count / counts.sum()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count} ({pct:.1f}%)",
                ha="center",
                va="bottom",
            )
        ax.set_title("Class balance")
        ax.set_ylabel("Patients")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "class_balance.png", dpi=150)
        plt.close(fig)

    def histograms(self, data) -> None:
        data[self.numeric].hist(figsize=(10, 8), bins=20, color="#4C78A8", edgecolor="white")
        plt.suptitle("Feature distributions")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "histograms.png", dpi=150)
        plt.close()

    def correlation_heatmap(self, data) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            data.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            ax=ax,
        )
        ax.set_title("Feature correlation")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "correlation_heatmap.png", dpi=150)
        plt.close(fig)

    def train_test_class_balance(self, train, test, target: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, split, name in ((axes[0], train, "Train"), (axes[1], test, "Test")):
            counts = split[target].value_counts().sort_index()
            bars = ax.bar(["No disease", "Disease"], counts.values, color=["#4C78A8", "#E45756"])
            for bar, count in zip(bars, counts.values):
                pct = 100 * count / counts.sum()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{count} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                )
            ax.set_title(f"{name} class balance")
            ax.set_ylabel("Patients")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "train_test_class_balance.png", dpi=150)
        plt.close(fig)

    def train_test_distributions(self, train, test) -> None:
        columns = [column for column in self.numeric if column in train.columns]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, column in zip(axes.ravel(), columns):
            ax.hist(train[column], bins=20, alpha=0.6, label="Train", color="#4C78A8")
            ax.hist(test[column], bins=20, alpha=0.6, label="Test", color="#E45756")
            ax.set_title(column)
            ax.legend()
        fig.suptitle("Train vs test feature distributions")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "train_test_distributions.png", dpi=150)
        plt.close(fig)

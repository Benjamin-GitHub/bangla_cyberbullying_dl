# src/plot_training_curves.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_history(path: Path):
    """Load a Keras History.history dict saved as .npy."""
    return np.load(path, allow_pickle=True).item()


def plot_curves(history: dict, title_prefix: str, out_prefix: str):
    """Plot accuracy and loss curves (train vs val) and save as PNGs."""
    # Accuracy
    train_acc = history.get("accuracy")
    val_acc = history.get("val_accuracy")

    if train_acc is not None and val_acc is not None:
        epochs = range(1, len(train_acc) + 1)
        plt.figure()
        plt.plot(epochs, train_acc, label="Train accuracy")
        plt.plot(epochs, val_acc, label="Val accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} - Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        acc_path = PLOTS_DIR / f"{out_prefix}_accuracy.png"
        plt.savefig(acc_path)
        plt.close()

    # Loss
    train_loss = history.get("loss")
    val_loss = history.get("val_loss")

    if train_loss is not None and val_loss is not None:
        epochs = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train loss")
        plt.plot(epochs, val_loss, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix} – Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        loss_path = PLOTS_DIR / f"{out_prefix}_loss.png"
        plt.savefig(loss_path)
        plt.close()


def main():
    # Simple NN
    nn_hist_path = RESULTS_DIR / "simple_nn_history.npy"
    if nn_hist_path.exists():
        nn_history = load_history(nn_hist_path)
        plot_curves(
            nn_history,
            title_prefix="Simple NN (Embedding + GlobalAvgPool)",
            out_prefix="simple_nn",
        )

    # BiLSTM
    bilstm_hist_path = RESULTS_DIR / "bilstm_history.npy"
    if bilstm_hist_path.exists():
        bilstm_history = load_history(bilstm_hist_path)
        plot_curves(
            bilstm_history,
            title_prefix="BiLSTM",
            out_prefix="bilstm",
        )


if __name__ == "__main__":
    main()
# src/export_bilstm_to_h5.py
"""
Utility script to convert the existing BiLSTM model
from .keras format to a legacy .h5 file for deployment.

This DOES NOT retrain anything. It just:
  1) loads models/bilstm_bangla_cyberbullying.keras
  2) saves models/bilstm_bangla_cyberbullying.h5
"""

from pathlib import Path
import tensorflow as tf  # uses whatever TF version is in your local venv


# --- Paths (match your train_bilstm.py layout) -----------------------

MODEL_DIR = Path("models")
OLD_MODEL_PATH = MODEL_DIR / "bilstm_bangla_cyberbullying.keras"
NEW_MODEL_PATH = MODEL_DIR / "bilstm_bangla_cyberbullying.h5"


def main():
    if not OLD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Old model not found at: {OLD_MODEL_PATH}")

    print(f"Loading existing model from: {OLD_MODEL_PATH}")
    model = tf.keras.models.load_model(OLD_MODEL_PATH)

    # Save explicitly as HDF5 for better compatibility on HF (TF 2.20 + Keras 3)
    print(f"Saving model in HDF5 format to: {NEW_MODEL_PATH}")
    model.save(NEW_MODEL_PATH, save_format="h5")

    size_bytes = NEW_MODEL_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Done. New .h5 size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
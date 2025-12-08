# src/inference.py

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # pyright: ignore[reportMissingImports]

from src.data_utils import clean_text, ID2LABEL

# --------------------------- Paths & config ---------------------------

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "bilstm_bangla_cyberbullying.keras"
TOKENIZER_PATH = MODEL_DIR / "bilstm_tokenizer.joblib"
LABELS_PATH = MODEL_DIR / "bilstm_label_mapping.joblib"

# MUST match the value used in train_bilstm.py
MAX_SEQ_LEN = 128

# Lazy-loaded globals
_model: tf.keras.Model | None = None
_tokenizer = None
_id2label: Dict[int, str] | None = None


# -------------------------- Loading artefacts -------------------------

def load_artifacts() -> Tuple[tf.keras.Model, object, Dict[int, str]]:
    """
    Load the trained BiLSTM model, tokenizer, and id→label mapping.

    Uses module-level caching so they are only loaded once.
    """
    global _model, _tokenizer, _id2label

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _tokenizer is None:
        _tokenizer = joblib.load(TOKENIZER_PATH)

    if _id2label is None:
        # Prefer the saved mapping; fall back to ID2LABEL from data_utils
        try:
            _id2label = joblib.load(LABELS_PATH)
        except FileNotFoundError:
            _id2label = ID2LABEL

    return _model, _tokenizer, _id2label


# --------------------------- Core helpers -----------------------------

def _prepare_input(text: str, tokenizer) -> np.ndarray:
    """
    Clean the raw text and convert it into a padded sequence
    compatible with the BiLSTM model.
    """
    # Ensure string
    text = str(text)

    # Same cleaning as training
    cleaned = clean_text(text)

    # Tokenise and pad
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
    )
    return padded


# --------------------------- Public API -------------------------------

def predict_proba(text: str) -> Dict[str, float]:
    """
    Return a dict mapping label → probability for the given text.
    """
    model, tokenizer, id2label = load_artifacts()

    if not str(text).strip():
        # Empty input: return empty dict instead of calling the model
        return {}

    x = _prepare_input(text, tokenizer)

    # Model expects shape (1, MAX_SEQ_LEN)
    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)

    # Map index → label string
    out: Dict[str, float] = {}
    for idx, p in enumerate(probs):
        label = id2label.get(idx, f"class_{idx}")
        out[label] = float(p)

    return out


def predict_label(text: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict the most likely label for the given text.

    Returns:
        (best_label, best_probability, full_probabilities_dict)
    """
    probs = predict_proba(text)

    if not probs:
        raise ValueError("Cannot predict on empty text.")

    best_label = max(probs, key=probs.get)
    best_prob = probs[best_label]

    return best_label, best_prob, probs
# src/data_utils.py

import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from transformers import AutoTokenizer


# -------------------------------------------------------------------
#                       Text cleaning
# -------------------------------------------------------------------

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
USERNAME_PATTERN = re.compile(r"@\w+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Decorative / noisy separator patterns:
DECORATIVE_CHARS = "-_=+|<>{}•●♦♠♥★☝✔⚫৤❣॥—০0∆"
SEPARATOR_PATTERN = re.compile(rf"[{re.escape(DECORATIVE_CHARS)}]{{3,}}")
# Long repeats of the SAME non-alphanumeric character
REPEAT_NOISE_PATTERN = re.compile(r"([^A-Za-z0-9\u0980-\u09FF\s])\1{4,}")


def clean_text(text: str) -> str:
    """
    Basic text cleaning based on EDA findings:
    
    - Remove URLs (they are rarely informative for bullying).
    - Remove @usernames (can leak personal info, not needed for label).
    - Remove decorative separator patterns and long repeated symbols.
    - Preserve emojis and punctuation because they carry
      sentiment that may help classification.
    - Preserve numbers (e.g. years, counts) as they may appear in political content.
    - Normalise whitespace.
    """
    if not isinstance(text, str):
        text = str(text)

    text = URL_PATTERN.sub(" ", text)
    text = USERNAME_PATTERN.sub(" ", text)
    text = SEPARATOR_PATTERN.sub(" ", text)
    text = REPEAT_NOISE_PATTERN.sub(" ", text)
    text = text.strip()
    text = MULTI_SPACE_PATTERN.sub(" ", text)

    return text


# -------------------------------------------------------------------
#                       Label encoding
# -------------------------------------------------------------------

# Fixed mapping
LABEL2ID: Dict[str, int] = {
    "political": 0,
    "sexual": 1,
    "troll": 2,
    "threat": 3,
    "neutral": 4,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Map string labels to numeric ids."""
    # normalise to lower-case
    labels = labels.astype(str).str.lower()
    return labels.map(LABEL2ID).values


# -------------------------------------------------------------------
#       Core preparation: clean, filter, deduplicate, encode
# -------------------------------------------------------------------

def prepare_dataframe(
    df: pd.DataFrame,
    text_col: str = "Description",
    label_col: str = "Label"
) -> pd.DataFrame:
    """
    Select relevant columns, clean text, drop nulls and duplicates.
    """
    # Select relevant columns
    df = df[[text_col, label_col]].copy()

    # Normalise labels to lower-case strings
    df[label_col] = df[label_col].astype(str).str.lower()

    # Text cleaning
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    # Drop rows with empty text or null labels
    df[text_col].replace("", np.nan)
    df.dropna(subset=[text_col, label_col], inplace=True)

    # Drop exact duplicate entries (text + label)
    df.drop_duplicates(subset=[text_col, label_col], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# -------------------------------------------------------------------
#       Train / validation / test split with stratification
# -------------------------------------------------------------------

def stratified_splits(
    df: pd.DataFrame,
    text_col: str = "Description",
    label_col: str = "Label",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train/val/test splits.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert np.isclose(train_size + val_size + test_size, 1.0), \
        "train_size + val_size + test_size must equal 1.0"

    # Encode labels to ids
    y_all = encode_labels(df[label_col])
    X_all = df[text_col].values

    # First split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        test_size=(1.0 - train_size),
        stratify=y_all,
        random_state=random_state,
    )

    # Second split: val vs test from temp
    relative_test_size = test_size / (test_size + val_size)  # proportion within temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# -------------------------------------------------------------------
#               Keras tokenizer + padded sequences
# -------------------------------------------------------------------

def build_keras_tokenizer(
    texts: List[str],
    num_words: int = None,
    oov_token: str = "[OOV]"
):
    """
    Fit a Keras Tokenizer on the training texts.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(
    tokenizer,
    texts: List[str],
    max_len: int = 128 # Max length setted based on EDA ( Max ≈ 210 words, 95th percentile ≈ ~57 words)
) -> np.ndarray:
    """
    Convert a list/array of texts into padded sequences.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return padded


# -------------------------------------------------------------------
#       Transformer preparation (Hugging Face tokenizers)
# -------------------------------------------------------------------

def load_transformer_tokenizer(model_name: str = "xlm-roberta-base"):
    """
    Load a pretrained Hugging Face tokenizer.

    Based on EDA (Bangla + code-mixing), a multilingual subword model such as
    'xlm-roberta-base' or 'bert-base-multilingual-cased' is appropriate.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def encode_texts_transformer(
    tokenizer,
    texts: List[str],
    max_len: int = 128
) -> Dict[str, np.ndarray]:
    """
    Encode texts using a Hugging Face tokenizer.

    Returns numpy arrays:
        {
            "input_ids": shape (N, max_len),
            "attention_mask": shape (N, max_len)
        }
    """
    encodings = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors=None,  # will return lists
    )

    input_ids = np.array(encodings["input_ids"], dtype=np.int64)
    attention_mask = np.array(encodings["attention_mask"], dtype=np.int64)

    return {"input_ids": input_ids, "attention_mask": attention_mask}
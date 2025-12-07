# src/train_simple_nn.py

from pathlib import Path
from datetime import datetime

#import os
#import random
import numpy as np
import tensorflow as tf

#os.environ["PYTHONHASHSEED"] = "0"
#random.seed(0)
#np.random.seed(0)
#tf.random.set_seed(0)

import joblib
import pandas as pd
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

from src.data_utils import prepare_dataframe, stratified_splits, ID2LABEL

# -------------------------------------------------------------------
#                           Config
# -------------------------------------------------------------------

DATA_PATH = Path("data/CyberBulling_Dataset_Bangla.xlsx")
TEXT_COL = "Description"
LABEL_COL = "Label"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "simple_nn_results.txt"

TOKENIZER_PATH = MODEL_DIR / "simple_nn_tokenizer.joblib"
MODEL_PATH = MODEL_DIR / "simple_nn_bangla_cyberbullying.keras"
LABELS_PATH = MODEL_DIR / "simple_nn_label_mapping.joblib"

MAX_NUM_WORDS = 30000
MAX_SEQ_LEN = 128
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 15
PATIENCE = 3  # early stopping
# PATIENCE = 5  # early stopping on val_loss


# -------------------------------------------------------------------
#                       Data loading & prep
# -------------------------------------------------------------------

def load_data():
    """Read Excel, clean with prepare_dataframe, then stratified split."""
    df_raw = pd.read_excel(DATA_PATH)
    df = prepare_dataframe(df_raw, text_col=TEXT_COL, label_col=LABEL_COL)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(
        df, text_col=TEXT_COL, label_col=LABEL_COL
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tokenizer(texts, num_words=MAX_NUM_WORDS, oov_token="[OOV]"):
    """Fit a Keras Tokenizer on training texts only."""
    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token=oov_token,
        filters="",  # I already cleaned it before
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded(tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Convert texts → sequences → padded arrays."""
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return padded


# -------------------------------------------------------------------
#                       Build Keras model
# -------------------------------------------------------------------

def build_model(vocab_size, embed_dim, max_len, num_classes):
    inputs = layers.Input(shape=(max_len,), name="input_ids")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=max_len,
        name="embedding",
    )(inputs)

    # Simple, fast architecture
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    # x = layers.Dropout(0.4, name="dropout_1")(x) # I increased dropout to 0.4 to reduce overfitting
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="simple_text_classifier")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -------------------------------------------------------------------
#                           Main
# -------------------------------------------------------------------

def main():
    # 1. Load and split
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    num_classes = len(ID2LABEL)
    print(f"Number of classes: {num_classes}")

    # 2. Tokenizer + padded sequences
    print("Fitting tokenizer on training texts...")
    tokenizer = build_tokenizer(X_train.tolist(), num_words=MAX_NUM_WORDS)

    X_train_seq = texts_to_padded(tokenizer, X_train.tolist(), max_len=MAX_SEQ_LEN)
    X_val_seq = texts_to_padded(tokenizer, X_val.tolist(), max_len=MAX_SEQ_LEN)
    X_test_seq = texts_to_padded(tokenizer, X_test.tolist(), max_len=MAX_SEQ_LEN)

    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
    print(f"Effective vocab size: {vocab_size}")

    # 3. Build model
    model = build_model(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        max_len=MAX_SEQ_LEN,
        num_classes=num_classes,
    )
    model.summary()

    # 4. Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )

    # 5. Train
    history = model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )
    # Save training history for later visualisation
    history_path = RESULTS_DIR / "simple_nn_history.npy"
    np.save(history_path, history.history)

    # 6. Evaluate on validation & test sets
    print("\nEvaluating on validation set...")
    val_loss, val_acc = model.evaluate(X_val_seq, y_val, verbose=0)
    print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}")

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # 7. Detailed classification reports
    print("\nValidation classification report:")
    y_val_pred_probs = model.predict(X_val_seq)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    val_report = classification_report(
        y_val,
        y_val_pred,
        target_names=[ID2LABEL[i] for i in sorted(ID2LABEL.keys())],
    )
    print(val_report)
    val_cm = confusion_matrix(y_val, y_val_pred)
    print("Validation confusion matrix:")
    print(val_cm)

    print("\nTest classification report:")
    y_test_pred_probs = model.predict(X_test_seq)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)
    test_report = classification_report(
        y_test,
        y_test_pred,
        target_names=[ID2LABEL[i] for i in sorted(ID2LABEL.keys())],
    )
    print(test_report)
    test_cm = confusion_matrix(y_test, y_test_pred)
    print("Test confusion matrix:")
    print(test_cm)

    # 8. Append summary of this run to results file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}\n\n")
        f.write("Validation classification report:\n")
        f.write(val_report + "\n")
        f.write("Validation confusion matrix:\n")
        f.write(str(val_cm) + "\n\n")
        f.write("Test classification report:\n")
        f.write(test_report + "\n")
        f.write("Test confusion matrix:\n")
        f.write(str(test_cm) + "\n")

    # 8. Save tokenizer + label mapping for deployment
    joblib.dump(tokenizer, TOKENIZER_PATH)
    joblib.dump(ID2LABEL, LABELS_PATH)

    print(f"\nSaved Keras model to: {MODEL_PATH}")
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")
    print(f"Saved label mapping to: {LABELS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
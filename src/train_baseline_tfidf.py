# src/train_baseline_tfidf.py

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from src.data_utils import prepare_dataframe, stratified_splits, ID2LABEL

# -------------------------------------------------------------------
#                       Config
# -------------------------------------------------------------------

DATA_PATH = Path("data/CyberBulling_Dataset_Bangla.xlsx")
TEXT_COL = "Description"
LABEL_COL = "Label"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"
VECTORIZER_PATH = MODEL_DIR / "baseline_tfidf_vectorizer.joblib"
LABELS_PATH = MODEL_DIR / "label_mapping.joblib"


# -------------------------------------------------------------------
#                       Load & prepare data
# -------------------------------------------------------------------

def load_data():
    df_raw = pd.read_excel(DATA_PATH)
    df = prepare_dataframe(df_raw, text_col=TEXT_COL, label_col=LABEL_COL)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(
        df, text_col=TEXT_COL, label_col=LABEL_COL
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# -------------------------------------------------------------------
#                       Build baseline model
# -------------------------------------------------------------------

def build_baseline_pipeline():
    """
    TF-IDF + Logistic Regression classifier.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=30000,      # cap vocab size
        sublinear_tf=True,
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        multi_class="multinomial",
    )

    pipe = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )
    return pipe


def evaluate_split(name, y_true, y_pred):
    print(f"\n=== {name} performance ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[ID2LABEL[i] for i in sorted(ID2LABEL.keys())],
        )
    )
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    # 1. Load and split
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2. Build pipeline
    model = build_baseline_pipeline()

    # 3. Train on train set
    print("Fitting baseline TF-IDF + Logistic Regression model...")
    model.fit(X_train, y_train)

    # 4. Evaluate on validation and test sets
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    evaluate_split("Validation", y_val, y_val_pred)
    evaluate_split("Test", y_test, y_test_pred)

    # 5. Save model and components
    print(f"\nSaving full pipeline to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    # Optional: also store label mapping (ID2LABEL)
    joblib.dump(ID2LABEL, LABELS_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
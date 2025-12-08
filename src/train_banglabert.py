# src/train_banglabert.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
    get_linear_schedule_with_warmup,
)

from src.data_utils import prepare_dataframe, stratified_splits, ID2LABEL


# ----------------------------- Config ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "CyberBulling_Dataset_Bangla.xlsx"
TEXT_COL = "Description"
LABEL_COL = "Label"

# BanglaBERT base (ELECTRA-style)
MODEL_NAME = "csebuetnlp/banglabert" 
MAX_LENGTH = 128

OUTPUT_DIR = BASE_DIR / "models" / "banglabert_cyberbullying"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "banglabert_results.txt"

BATCH_SIZE = 8
NUM_EPOCHS = 3
SEED = 42


# -------------------------- Dataset class ------------------------------

class CyberbullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ------------------------- Data preparation ----------------------------

def load_splits():
    df_raw = pd.read_excel(DATA_PATH)
    df = prepare_dataframe(df_raw, text_col=TEXT_COL, label_col=LABEL_COL)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(
        df, text_col=TEXT_COL, label_col=LABEL_COL
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------- Metrics -----------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def evaluate_model(model, data_loader, device, target_names):
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    logits = np.concatenate(all_logits)

    metrics = compute_metrics((logits, y_true))
    report = classification_report(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred)

    return metrics, report, cm, y_true, y_pred


# ------------------------------ Main -----------------------------------

def main():
    set_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    num_labels = len(ID2LABEL)
    id2label = {i: ID2LABEL[i] for i in ID2LABEL}
    label2id = {v: k for k, v in ID2LABEL.items()}

    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]

    print("Loading tokenizer and model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model.to(device)

    # Datasets
    train_dataset = CyberbullyingDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = CyberbullyingDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = CyberbullyingDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_val_macro_f1 = -1.0
    best_val_metrics = None

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k != "labels"}

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_metrics, val_report, val_cm, _, _ = evaluate_model(
            model, val_loader, device, target_names
        )
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Val macro F1: {val_macro_f1:.4f}"
        )

        # Save best model based on validation macro F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_metrics = val_metrics
            model.save_pretrained(str(OUTPUT_DIR))
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            print("Saved new best model.")

    # Load best model for test evaluation
    print("Loading best model from disk...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(OUTPUT_DIR),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    best_model.to(device)

    print("Evaluating on validation set (best model)...")
    val_metrics, val_report, val_cm, _, _ = evaluate_model(
        best_model, val_loader, device, target_names
    )
    print(val_metrics)
    print("\nValidation classification report:")
    print(val_report)
    print("Validation confusion matrix:")
    print(val_cm)

    print("Evaluating on test set...")
    test_metrics, test_report, test_cm, y_test_true, y_test_pred = evaluate_model(
        best_model, test_loader, device, target_names
    )
    test_acc = test_metrics["accuracy"]
    test_macro_f1 = test_metrics["macro_f1"]
    test_weighted_f1 = test_metrics["weighted_f1"]

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Test weighted F1: {test_weighted_f1:.4f}")

    print("\nTest classification report:")
    print(test_report)
    print("Test confusion matrix:")
    print(test_cm)

    # Log results to text file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Val metrics (best): {val_metrics}\n")
        f.write(
            f"Test accuracy: {test_acc:.4f}, "
            f"macro F1: {test_macro_f1:.4f}, "
            f"weighted F1: {test_weighted_f1:.4f}\n\n"
        )
        f.write("Validation classification report:\n")
        f.write(val_report + "\n")
        f.write("Validation confusion matrix:\n")
        f.write(str(val_cm) + "\n\n")
        f.write("Test classification report:\n")
        f.write(test_report + "\n")
        f.write("Test confusion matrix:\n")
        f.write(str(test_cm) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
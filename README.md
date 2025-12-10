# 🛡️ Bangla Cyberbullying Detection (Deep Learning Project)

This repository contains the development code for a **Bangla cyberbullying detection** system, created for the LSBU module **CSI_7_DEL – Deep Learning**.

The goal is to classify Bangla social media comments into **five categories**:

- `Political`
- `Sexual`
- `Troll`
- `Threat`
- `Neutral`

The repo includes several models (baseline + deep learning) and a Streamlit app used for interactive inference and deployment.

---

## 📂 Project Structure

```text
bangla_cyberbullying_dl/
├── app.py                      # Streamlit app (local / HF Spaces interface)
├── requirements.txt            # Python dependencies
├── readme.md                   # This file (development README)
├── LICENSE                     # Project license
├── data/                       # (Not tracked) raw / processed data
│   └── CyberBulling_Dataset_Bangla.xlsx   # Main labelled dataset (local only)
├── models/
│   ├── banglabert_cyberbullying/         # Fine‑tuned BanglaBERT model (HF format)
│   ├── bilstm_bangla_cyberbullying.*     # Trained BiLSTM weights (.keras / .h5)
│   ├── simple_nn_bangla_cyberbullying.*  # Trained simple NN model
│   ├── bilstm_label_mapping.joblib       # Label → index mapping for BiLSTM
│   ├── simple_nn_label_mapping.joblib    # Label → index mapping for simple NN
│   └── label_mapping.joblib              # Shared mapping (for legacy scripts)
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis notebook
├── results/
│   ├── bilstm_results*.txt      # BiLSTM evaluation summaries
│   ├── simple_nn_results*.txt   # Simple NN evaluation summaries
│   ├── banglabert_results.txt   # BanglaBERT evaluation summary
│   ├── bilstm_history.npy       # Keras training history (BiLSTM)
│   ├── simple_nn_history.npy    # Keras training history (simple NN)
│   └── plots/                   # Generated plots (training curves, etc.)
└── src/
    ├── __init__.py
    ├── config.py                # Central config (paths, hyper‑parameters)
    ├── data_utils.py            # Data loading, cleaning, splitting utilities
    ├── export_bilstm_to_h5.py   # Script to export BiLSTM to .h5 for deployment
    ├── inference.py             # `predict_label` + model loading helpers
    ├── plot_training_curves.py  # Plot loss/accuracy curves from history files
    ├── train_baseline_tfidf.py  # Baseline TF‑IDF + linear model
    ├── train_simple_nn.py       # Simple feed‑forward neural network
    ├── train_bilstm.py          # BiLSTM model training
    └── train_banglabert.py      # BanglaBERT / transformer fine‑tuning
```

> Note: the **data** and **models** directories are intentionally not tracked by git (or are handled via Git LFS on the deployment repo) because of file size and privacy.

---

## 📊 Dataset

The dataset consists of Bangla social media comments collected from platforms such as **YouTube, Facebook, and Twitter/X**. Each comment is annotated with one of the five cyberbullying categories listed above.

Typical dataset format (Excel):

- `Description` – Bangla comment text
- `Label` – one of `{political, sexual, troll, threat, neutral}`

The main dataset file is expected at:

```text
data/CyberBulling_Dataset_Bangla.xlsx
```

This file is **not** included in the repository and must be placed manually in the `data/` directory.

---

## 🧹 Data Preparation

All data preparation logic is centralised in `src/data_utils.py` so that training scripts, notebooks, and the Streamlit app all share the same preprocessing steps.

Key steps:

1. **Column selection**  
   - Drop non‑semantic index columns (e.g. `Unnamed: 0`).  
   - Keep `Description` (input text) and `Label` (target).

2. **Text cleaning** (applied to each comment):
   - Remove URLs (`http://`, `https://`, `www.`).
   - Remove user mentions (`@username`).
   - Normalise whitespace and strip leading/trailing spaces.
   - Optionally handle emojis / non‑Bangla characters depending on the model.

3. **Filtering & deduplication**  
   - Drop rows with empty or missing text.  
   - Drop rows with missing labels.  
   - Remove exact duplicate rows.

4. **Label encoding**  
   - Create a mapping between text labels and integer indices, saved to `*.joblib` files for reuse at inference time.

5. **Train / validation / test split**  
   - Stratified split to preserve label proportions (e.g. 70/15/15).  
   - Optionally support custom splits via `config.py`.

6. **Input representations**  
   - For classical / deep models (simple NN, BiLSTM): tokenise and pad sequences.  
   - For transformers (BanglaBERT): use a Hugging Face tokenizer to create `input_ids` and `attention_mask` tensors.

---

## 🧠 Models

This repo includes several modelling approaches used in the coursework experiments:

1. **Baseline – TF‑IDF + Linear Classifier**  
   Implemented in `src/train_baseline_tfidf.py` using scikit‑learn. Serves as a classical ML baseline.

2. **Simple Neural Network (MLP)**  
   Implemented in `src/train_simple_nn.py`. Uses an embedding + averaged representation (or TF‑IDF) followed by dense layers for 5‑way classification.

3. **BiLSTM (Main Deployed DL Model)**  
   Implemented in `src/train_bilstm.py`. Architecture typically includes:
   - Embedding layer (random or pre‑trained)  
   - Bidirectional LSTM layer(s)  
   - Dense layers with dropout  
   - Softmax output over the 5 classes

   Trained weights are saved under `models/bilstm_bangla_cyberbullying.*` and are loaded by the Streamlit app through `src.inference.py`.

4. **BanglaBERT / Transformer (Advanced Model)**  
   Implemented in `src/train_banglabert.py` using Hugging Face `transformers`.  
   This model is more computationally expensive and mainly used for comparison and analysis. The fine‑tuned model is stored in `models/banglabert_cyberbullying/`.

---

## 📈 Evaluation & Results

Evaluation artefacts are saved in the `results/` folder:

- `*_results.txt` – summary metrics (accuracy, precision, recall, F1, etc.).
- `*_history.npy` – Keras training history objects (loss/accuracy per epoch).
- `results/plots/` – figures produced by `src/plot_training_curves.py`.

These files were used to compare models in the written report and to justify the choice of BiLSTM as the main deployed model.

---

## 🚀 Training the Models

Before training, ensure the virtual environment is active and the dataset is in `data/`.

### 1. Baseline (TF‑IDF)

```bash
python -m src.train_baseline_tfidf
```

### 2. Simple Neural Network

```bash
python -m src.train_simple_nn
```

### 3. BiLSTM

```bash
python -m src.train_bilstm
```

### 4. BanglaBERT (Optional, GPU recommended)

```bash
python -m src.train_banglabert
```

Training scripts will automatically write results to the `results/` directory and save models under `models/` (paths and hyper‑parameters are controlled from `src/config.py`).

---

## 🌐 Streamlit App (Local / Hugging Face)

The interactive interface is implemented in `app.py` and uses the **BiLSTM** model via the helper functions in `src/inference.py`.

### Run locally

```bash
streamlit run app.py
```

The app supports:

- Free‑text input for a single Bangla comment.
- A small list of built‑in demo examples (one per class).
- Display of predicted label and class probabilities.

### Deploy to Hugging Face Spaces

The same `app.py` can be used as the entry point for a Hugging Face Space:

1. Create a new Space (SDK = Streamlit).
2. Push this repo (or a deployment‑only copy) to the Space.
3. Upload the trained BiLSTM model files to the `models/` folder in the Space.
4. Ensure `requirements.txt` matches the versions used during development.

The Space will automatically start the Streamlit app using `app.py`.

---

## 🔧 Installation (Development Environment)

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/bangla_cyberbullying_dl.git
cd bangla_cyberbullying_dl
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Add local data and models**

- Place the Excel dataset in `data/`.  
- Place trained model files (if already trained) in `models/`.  
- Alternatively, run the training scripts to regenerate models.

---

## 🧾 Coursework Mapping (CSI_7_DEL)

This development repo underpins the written deep learning coursework:

- **Data Understanding & Preparation** – implemented mainly in `src/data_utils.py` and `notebooks/eda.ipynb`.
- **Modelling & Evaluation** – implemented in the various `train_*.py` scripts and stored in `results/`.
- **Deployment** – implemented through `app.py` (local Streamlit + Hugging Face Space).

The README is focused on the **developer view** of the project so the codebase can be understood and reused later.

---

## © Copyright

© 2025 Benjamin Mehrdad. All rights reserved.

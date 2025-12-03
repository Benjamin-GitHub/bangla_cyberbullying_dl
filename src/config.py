import os
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "CyberBulling_Dataset_Bangla.xlsx"

# Directories for artefacts
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

# Make sure dirs exist (safe to call multiple times)
for d in [DATA_DIR, NOTEBOOKS_DIR, SAVED_MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
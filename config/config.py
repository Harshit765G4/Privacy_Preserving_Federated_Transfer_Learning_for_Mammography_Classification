"""
======================================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

File:
config.py

Purpose:
Central configuration file used across the entire project.

Author:
Harshit Garg
======================================================================
"""

from pathlib import Path
import torch

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"

LOGS_DIR = PROJECT_ROOT / "logs"

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

# Automatically create folders if missing

CHECKPOINT_DIR.mkdir(exist_ok=True)

RESULTS_DIR.mkdir(exist_ok=True)

LOGS_DIR.mkdir(exist_ok=True)

EXPERIMENT_DIR.mkdir(exist_ok=True)

# ============================================================
# DATASET FILES
# ============================================================

DATASET_CSV = PROCESSED_DATA_DIR / "mammogram_dataset_with_folds.csv"

# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 512

NUM_CHANNELS = 3

NUM_CLASSES = 2

# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 4

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# ============================================================
# CROSS VALIDATION
# ============================================================

N_FOLDS = 5

CURRENT_FOLD = 0

# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING_PATIENCE = 7

# ============================================================
# RANDOMNESS
# ============================================================

SEED = 42

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "ResNet50"

PRETRAINED = True

# ============================================================
# CHECKPOINTS
# ============================================================

BEST_MODEL = CHECKPOINT_DIR / "best_resnet50.pth"

LAST_MODEL = CHECKPOINT_DIR / "last_resnet50.pth"

# ============================================================
# LOGGING
# ============================================================

PRINT_EVERY = 10

SAVE_BEST_ONLY = True

# ============================================================
# FEDERATED LEARNING (Future)
# ============================================================

NUM_CLIENTS = 3

LOCAL_EPOCHS = 2

ROUNDS = 20

# ============================================================
# DIFFERENTIAL PRIVACY (Future)
# ============================================================

TARGET_EPSILON = 1.9

TARGET_DELTA = 1e-5

MAX_GRAD_NORM = 1.0
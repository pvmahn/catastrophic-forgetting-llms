import torch
import os

# ─── MODEL ───────────────────────────────────────────
MODEL_NAME = "gpt2-medium"

# ─── DEVICE ──────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── DATASET ─────────────────────────────────────────
TASKS = ["hellaswag", "medmcqa", "sst2"]
TASK_NAMES = {
    "hellaswag": "Commonsense Reasoning",
    "medmcqa"  : "Medical QA",
    "sst2"     : "Sentiment Analysis",
}

# ─── TỰ ĐỘNG DETECT KAGGLE HAY LOCAL ─────────────────
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    MAX_TRAIN_SAMPLES = 10000
    MAX_EVAL_SAMPLES  = 1000
    BATCH_SIZE        = 8
    MODEL_DIR         = "/kaggle/working/models"
    RESULTS_DIR       = "/kaggle/working/results"
    FIGURES_DIR       = "/kaggle/working/figures"
    DATA_DIR          = "/kaggle/working/data"
else:
    MAX_TRAIN_SAMPLES = 3000
    MAX_EVAL_SAMPLES  = 500
    BATCH_SIZE        = 2
    MODEL_DIR         = "models"
    RESULTS_DIR       = "results"
    FIGURES_DIR       = "figures"
    DATA_DIR          = "data"

# Tạo thư mục nếu chưa có
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ─── TRAINING ────────────────────────────────────────
EPOCHS     = 3
LR         = 2e-5
MAX_LENGTH = 128
FP16       = True

# ─── PATHS ───────────────────────────────────────────
CHECKPOINT = {
    "hellaswag": f"{MODEL_DIR}/checkpoint_task1",
    "medmcqa"  : f"{MODEL_DIR}/checkpoint_task2",
    "sst2"     : f"{MODEL_DIR}/checkpoint_task3",
}

CHECKPOINT_EWC = {
    "hellaswag": f"{MODEL_DIR}/checkpoint_ewc_task1",
    "medmcqa"  : f"{MODEL_DIR}/checkpoint_ewc_task2",
    "sst2"     : f"{MODEL_DIR}/checkpoint_ewc_task3",
}

CHECKPOINT_LORA = {
    "hellaswag": f"{MODEL_DIR}/checkpoint_lora_task1",
    "medmcqa"  : f"{MODEL_DIR}/checkpoint_lora_task2",
    "sst2"     : f"{MODEL_DIR}/checkpoint_lora_task3",
}

CHECKPOINT_QLORA = {
    "hellaswag": f"{MODEL_DIR}/checkpoint_qlora_task1",
    "medmcqa"  : f"{MODEL_DIR}/checkpoint_qlora_task2",
    "sst2"     : f"{MODEL_DIR}/checkpoint_qlora_task3",
}
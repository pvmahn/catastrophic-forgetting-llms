import os
import torch
import pandas as pd
from config import *
from dataset_loader import get_tokenizer
from train import train_task, load_model
from evaluate import (evaluate_all_tasks, load_model_from_checkpoint,
                      save_results, compute_forgetting)
from plot import plot_all


def main():
    print("=" * 60)
    print("  SEQUENTIAL FINE-TUNING — CATASTROPHIC FORGETTING")
    print("=" * 60)
    print(f"\n  Device : {DEVICE}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Tasks  : {' → '.join([TASK_NAMES[t] for t in TASKS])}")
    print(f"  Epochs : {EPOCHS}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"  Train  : {MAX_TRAIN_SAMPLES} samples/task")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    tokenizer      = get_tokenizer()
    results_matrix = {}

    # ──────────────────────────────────────────────────────
    # BƯỚC 1: Fine-tune Task 1 — HellaSwag
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 1: Fine-tune Task 1 — Commonsense Reasoning")
    print("─" * 60)

    model = train_task("hellaswag", from_checkpoint=None)

    print("\n Đánh giá sau Task 1:")
    results_matrix["after_task1"] = evaluate_all_tasks(
        model, ["hellaswag"], tokenizer
    )

    del model
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────
    # BƯỚC 2: Fine-tune Task 2 — MedMCQA
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 2: Fine-tune Task 2 — Medical QA")
    print("  (Tiếp tục từ checkpoint Task 1)")
    print("─" * 60)

    model = train_task("medmcqa",
                       from_checkpoint=CHECKPOINT["hellaswag"])

    print("\n Đánh giá sau Task 2:")
    results_matrix["after_task2"] = evaluate_all_tasks(
        model, ["hellaswag", "medmcqa"], tokenizer
    )

    del model
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────
    # BƯỚC 3: Fine-tune Task 3 — SST-2
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 3: Fine-tune Task 3 — Sentiment Analysis")
    print("  (Tiếp tục từ checkpoint Task 2)")
    print("─" * 60)

    model = train_task("sst2",
                       from_checkpoint=CHECKPOINT["medmcqa"])

    print("\n Đánh giá sau Task 3:")
    results_matrix["after_task3"] = evaluate_all_tasks(
        model, ["hellaswag", "medmcqa", "sst2"], tokenizer
    )

    del model
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────
    # BƯỚC 4: Tính Forgetting Score & Lưu kết quả
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 4: Tổng hợp kết quả Sequential")
    print("─" * 60)

    forgetting_scores = compute_forgetting(results_matrix)
    df                = save_results(results_matrix, prefix="seq_")

    print("\n BẢNG KẾT QUẢ ACCURACY (%):")
    print(df.to_string(index=False))

    print("\n FORGETTING SCORES:")
    for k, v in forgetting_scores.items():
        print(f"  {k}: {v}%")

    # ──────────────────────────────────────────────────────
    # BƯỚC 5: Vẽ biểu đồ
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 5: Vẽ biểu đồ Sequential")
    print("─" * 60)

    plot_all(results_matrix, forgetting_scores)

    print("\n" + "=" * 60)
    print(" HOÀN THÀNH SEQUENTIAL FINE-TUNING!")
    print(f" Kết quả lưu tại : {RESULTS_DIR}/")
    print(f" Biểu đồ lưu tại : {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
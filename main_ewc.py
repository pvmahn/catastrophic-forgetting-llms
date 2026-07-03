import os
import gc
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import *
from dataset_loader import get_tokenizer, get_dataset
from train_ewc import EWC, train_task_ewc
from train import load_model
from evaluate import (evaluate_all_tasks, save_results, compute_forgetting)
from plot import plot_all_ewc


def main_ewc():
    print("=" * 60)
    print("  EWC FINE-TUNING — CATASTROPHIC FORGETTING")
    print("=" * 60)
    print(f"\n  Device : {DEVICE}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Lambda : 5000")
    print(f"  Train  : {MAX_TRAIN_SAMPLES} samples/task")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    tokenizer  = get_tokenizer()
    results_ewc = {}
    ewc_list   = []
    EWC_LAMBDA = 5000

    # ──────────────────────────────────────────────
    # BƯỚC 1: EWC Task 1 — HellaSwag
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 1: EWC Task 1 — Commonsense Reasoning")
    print("─" * 60)

    model, train_loader = train_task_ewc(
        "hellaswag",
        from_checkpoint=None,
        ewc_list=[],
        ewc_lambda=EWC_LAMBDA
    )

    print("\nTinh EWC cho Task 1...")
    ewc1 = EWC(model, train_loader, DEVICE)
    ewc_list.append(ewc1)

    print("\nDanh gia sau Task 1 (EWC):")
    results_ewc["after_task1"] = evaluate_all_tasks(
        model, ["hellaswag"], tokenizer
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # BƯỚC 2: EWC Task 2 — MedMCQA
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 2: EWC Task 2 — Medical QA")
    print("─" * 60)

    model, train_loader = train_task_ewc(
        "medmcqa",
        from_checkpoint=CHECKPOINT_EWC["hellaswag"],
        ewc_list=ewc_list,
        ewc_lambda=EWC_LAMBDA
    )

    print("\nTinh EWC cho Task 2...")
    ewc2 = EWC(model, train_loader, DEVICE)
    ewc_list.append(ewc2)

    print("\nDanh gia sau Task 2 (EWC):")
    results_ewc["after_task2"] = evaluate_all_tasks(
        model, ["hellaswag", "medmcqa"], tokenizer
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ──────────────────────────────────────────────
    # BƯỚC 3: EWC Task 3 — SST-2
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 3: EWC Task 3 — Sentiment Analysis")
    print("─" * 60)

    model, train_loader = train_task_ewc(
        "sst2",
        from_checkpoint=CHECKPOINT_EWC["medmcqa"],
        ewc_list=ewc_list,
        ewc_lambda=EWC_LAMBDA
    )

    print("\nDanh gia sau Task 3 (EWC):")
    results_ewc["after_task3"] = evaluate_all_tasks(
        model, ["hellaswag", "medmcqa", "sst2"], tokenizer
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # BƯỚC 4: Tính Forgetting + Lưu kết quả
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 4: Tổng hợp kết quả EWC")
    print("─" * 60)

    forgetting_ewc = compute_forgetting(results_ewc)
    df             = save_results(results_ewc, prefix="ewc_")

    print("\n BẢNG KẾT QUẢ EWC (%):")
    print(df.to_string(index=False))

    print("\n FORGETTING SCORES EWC:")
    for k, v in forgetting_ewc.items():
        print(f"  {k}: {v}%")

    # Lưu CSV so sánh (cần kết quả Sequential từ main.py)
    seq_csv = f"{RESULTS_DIR}/seq_forgetting_scores.csv"
    if os.path.exists(seq_csv):
        seq_df = pd.read_csv(seq_csv)
        tasks_cf = ["Forgetting HellaSwag (%)", "Forgetting MedMCQA (%)"]
        forgetting_seq = {
            row.iloc[0]: row.iloc[1]
            for _, row in seq_df.iterrows()
        }
        df_compare = pd.DataFrame({
            "Task"      : tasks_cf,
            "Sequential": [forgetting_seq.get(t, "-") for t in tasks_cf],
            "EWC"       : [forgetting_ewc.get(t, "-") for t in tasks_cf],
        })
        df_compare.to_csv(f"{RESULTS_DIR}/comparison_seq_vs_ewc.csv",
                          index=False)
        print(f"\nDa luu: {RESULTS_DIR}/comparison_seq_vs_ewc.csv")

    # ──────────────────────────────────────────────
    # BƯỚC 5: Vẽ biểu đồ EWC
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 5: Vẽ biểu đồ EWC")
    print("─" * 60)

    plot_all_ewc(results_ewc, forgetting_ewc)

    print("\n" + "=" * 60)
    print(" HOÀN THÀNH EWC FINE-TUNING!")
    print(f" Kết quả lưu tại : {RESULTS_DIR}/")
    print(f" Biểu đồ lưu tại : {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main_ewc()
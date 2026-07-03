import os
import gc
import torch
import pandas as pd
from config import *
from dataset_loader import get_tokenizer
from train_qlora import train_task_qlora
from evaluate import (evaluate_all_tasks, save_results, compute_forgetting)
from plot import plot_all_lora


def main_qlora():
    print("=" * 60)
    print("  QLoRA FINE-TUNING — CATASTROPHIC FORGETTING")
    print("=" * 60)
    print(f"\n  Device : {DEVICE}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  QLoRA r: 16")
    print(f"  Quant  : 4-bit NF4")
    print(f"  Train  : {MAX_TRAIN_SAMPLES} samples/task")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    tokenizer     = get_tokenizer()
    results_qlora = {}

    # ──────────────────────────────────────────────
    # BƯỚC 1: QLoRA Task 1 — HellaSwag
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 1: QLoRA Task 1 — Commonsense Reasoning")
    print("─" * 60)

    model = train_task_qlora("hellaswag", from_checkpoint=None)

    print("\n Đánh giá sau Task 1 (QLoRA):")
    results_qlora["after_task1"] = evaluate_all_tasks(
        model, ["hellaswag"], tokenizer
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # BƯỚC 2: QLoRA Task 2 — MedMCQA
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 2: QLoRA Task 2 — Medical QA")
    print("  (Tiếp tục từ QLoRA checkpoint Task 1)")
    print("─" * 60)

    model = train_task_qlora(
        "medmcqa",
        from_checkpoint=CHECKPOINT_QLORA["hellaswag"]
    )

    print("\n Đánh giá sau Task 2 (QLoRA):")
    results_qlora["after_task2"] = evaluate_all_tasks(
        model, ["hellaswag", "medmcqa"], tokenizer
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # BƯỚC 3: QLoRA Task 3 — SST-2
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 3: QLoRA Task 3 — Sentiment Analysis")
    print("  (Tiếp tục từ QLoRA checkpoint Task 2)")
    print("─" * 60)

    model = train_task_qlora(
        "sst2",
        from_checkpoint=CHECKPOINT_QLORA["medmcqa"]
    )

    print("\n Đánh giá sau Task 3 (QLoRA):")
    results_qlora["after_task3"] = evaluate_all_tasks(
        model, ["hellaswag", "medmcqa", "sst2"], tokenizer
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # BƯỚC 4: Tính Forgetting + Lưu kết quả
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 4: Tổng hợp kết quả QLoRA")
    print("─" * 60)

    forgetting_qlora = compute_forgetting(results_qlora)
    df               = save_results(results_qlora, prefix="qlora_")

    print("\n BẢNG KẾT QUẢ QLoRA (%):")
    print(df.to_string(index=False))

    print("\n FORGETTING SCORES QLoRA:")
    for k, v in forgetting_qlora.items():
        print(f"  {k}: {v}%")

    _compare_3_methods(forgetting_qlora)

    # ──────────────────────────────────────────────
    # BƯỚC 5: Vẽ biểu đồ QLoRA
    # ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" BƯỚC 5: Vẽ biểu đồ QLoRA")
    print("─" * 60)

    plot_all_lora(results_qlora, forgetting_qlora)

    print("\n" + "=" * 60)
    print(" HOÀN THÀNH QLoRA FINE-TUNING!")
    print(f" Kết quả lưu tại : {RESULTS_DIR}/")
    print(f" Biểu đồ lưu tại : {FIGURES_DIR}/")
    print("=" * 60)


def _compare_3_methods(forgetting_qlora):
    seq_csv = f"{RESULTS_DIR}/seq_forgetting_scores.csv"
    ewc_csv = f"{RESULTS_DIR}/ewc_forgetting_scores.csv"

    if not os.path.exists(seq_csv) or not os.path.exists(ewc_csv):
        print("\n⚠️ Chua co ket qua Sequential hoac EWC!")
        print("   Chay main.py va main_ewc.py truoc de so sanh!")
        return

    seq_df = pd.read_csv(seq_csv)
    ewc_df = pd.read_csv(ewc_csv)

    forgetting_seq = dict(zip(seq_df.iloc[:, 0], seq_df.iloc[:, 1]))
    forgetting_ewc = dict(zip(ewc_df.iloc[:, 0], ewc_df.iloc[:, 1]))

    tasks_cf = ["Forgetting HellaSwag (%)", "Forgetting MedMCQA (%)"]

    print("\n" + "=" * 65)
    print(" BANG SO SANH 3 PHUONG PHAP")
    print("=" * 65)
    print(f"{'Task':<28} {'Sequential':>12} {'EWC':>12} {'QLoRA':>12}")
    print("-" * 65)
    for t in tasks_cf:
        seq_val   = forgetting_seq.get(t, "-")
        ewc_val   = forgetting_ewc.get(t, "-")
        qlora_val = forgetting_qlora.get(t, "-")
        print(f"{t:<28} {seq_val:>11.1f}% {ewc_val:>11.1f}% {qlora_val:>11.1f}%")
    print("=" * 65)

    df_compare = pd.DataFrame({
        "Task"      : tasks_cf,
        "Sequential": [forgetting_seq.get(t, "-")   for t in tasks_cf],
        "EWC"       : [forgetting_ewc.get(t, "-")   for t in tasks_cf],
        "QLoRA"     : [forgetting_qlora.get(t, "-") for t in tasks_cf],
    })
    path = f"{RESULTS_DIR}/comparison_3methods.csv"
    df_compare.to_csv(path, index=False)
    print(f"\n Da luu: {path}")

    from plot import plot_all_comparison

    seq_acc_csv   = f"{RESULTS_DIR}/seq_accuracy_table.csv"
    ewc_acc_csv   = f"{RESULTS_DIR}/ewc_accuracy_table.csv"
    qlora_acc_csv = f"{RESULTS_DIR}/qlora_accuracy_table.csv"

    if (os.path.exists(seq_acc_csv) and
        os.path.exists(ewc_acc_csv) and
        os.path.exists(qlora_acc_csv)):

        def csv_to_matrix(path):
            df = pd.read_csv(path)
            matrix = {}
            stages = ["after_task1", "after_task2", "after_task3"]
            task_cols = {
                "Commonsense Reasoning": "hellaswag",
                "Medical QA"           : "medmcqa",
                "Sentiment Analysis"   : "sst2"
            }
            for i, stage in enumerate(stages):
                row = df.iloc[i]
                matrix[stage] = {}
                for col, task in task_cols.items():
                    val = row.get(col, None)
                    try:
                        matrix[stage][task] = float(
                            str(val).replace("%", "").strip()
                        ) if val != "-" else None
                    except:
                        matrix[stage][task] = None
            return matrix

        seq_matrix   = csv_to_matrix(seq_acc_csv)
        ewc_matrix   = csv_to_matrix(ewc_acc_csv)
        qlora_matrix = csv_to_matrix(qlora_acc_csv)

        plot_all_comparison(
            forgetting_seq, forgetting_ewc, forgetting_qlora,
            seq_matrix, ewc_matrix, qlora_matrix
        )
        print(" Da ve bieu do so sanh 3 phuong phap!")


if __name__ == "__main__":
    main_qlora()
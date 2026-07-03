import torch
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification
from tqdm import tqdm
import pandas as pd
import os
from config import *
from dataset_loader import get_dataset, get_tokenizer, get_num_labels


# ─── EVALUATE 1 TASK ─────────────────────────────────
def evaluate_one_task(model, task_name, tokenizer):
    _, eval_dataset = get_dataset(task_name, tokenizer)
    eval_loader     = DataLoader(eval_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"  Evaluating {task_name}"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            preds    = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return round(accuracy * 100, 2)


# ─── EVALUATE TẤT CẢ TASK ────────────────────────────
def evaluate_all_tasks(model, tasks_to_eval, tokenizer):
    results = {}
    for task in tasks_to_eval:
        print(f"\n Đánh giá task: {TASK_NAMES[task]}")
        acc = evaluate_one_task(model, task, tokenizer)
        results[task] = acc
        print(f"   Accuracy: {acc}%")
    return results


# ─── LOAD MODEL TỪ CHECKPOINT ────────────────────────
def load_model_from_checkpoint(checkpoint_path, task_name):
    num_labels = get_num_labels(task_name)
    model      = GPT2ForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.config.pad_token_id = model.config.eos_token_id
    model = model.to(DEVICE)
    return model


# ─── LƯU KẾT QUẢ ─────────────────────────────────────
def save_results(results_matrix, prefix=""):
    """
    prefix: "" = Sequential, "ewc_" = EWC, "lora_" = LoRA
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for after_task, task_scores in results_matrix.items():
        row = {"Sau khi học": after_task}
        for task in TASKS:
            row[TASK_NAMES[task]] = task_scores.get(task, "-")
        rows.append(row)

    df = pd.DataFrame(rows)
    fname = f"{RESULTS_DIR}/{prefix}accuracy_table.csv"
    df.to_csv(fname, index=False)
    print(f"\n💾 Đã lưu bảng kết quả tại: {fname}")

    forgetting = compute_forgetting(results_matrix)
    fg_df      = pd.DataFrame([forgetting])
    fg_fname   = f"{RESULTS_DIR}/{prefix}forgetting_scores.csv"
    fg_df.to_csv(fg_fname, index=False)
    print(f" Đã lưu forgetting scores tại: {fg_fname}")

    return df


# ─── TÍNH FORGETTING SCORE ───────────────────────────
def compute_forgetting(results_matrix):
    forgetting = {}

    if "after_task1" in results_matrix and "after_task3" in results_matrix:
        peak  = results_matrix["after_task1"].get("hellaswag", 0)
        final = results_matrix["after_task3"].get("hellaswag", 0)
        forgetting["Forgetting HellaSwag (%)"] = round(peak - final, 2)

    if "after_task2" in results_matrix and "after_task3" in results_matrix:
        peak  = results_matrix["after_task2"].get("medmcqa", 0)
        final = results_matrix["after_task3"].get("medmcqa", 0)
        forgetting["Forgetting MedMCQA (%)"] = round(peak - final, 2)

    print(f"\n Forgetting Scores: {forgetting}")
    return forgetting


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    ckpt = CHECKPOINT["hellaswag"]
    if os.path.exists(ckpt):
        model   = load_model_from_checkpoint(ckpt, "hellaswag")
        results = evaluate_all_tasks(model, ["hellaswag"], tokenizer)
        print(results)
    else:
        print("  Chưa có checkpoint, hãy chạy train.py trước!")
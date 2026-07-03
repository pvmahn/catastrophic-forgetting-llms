# train.py
# Fine-tune GPT2 trên 1 task cụ thể

import torch
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from config import *
from dataset_loader import get_dataset, get_tokenizer, get_num_labels


# ─── LOAD MODEL ──────────────────────────────────────
def load_model(task_name, from_checkpoint=None):
    num_labels = get_num_labels(task_name)

    if from_checkpoint and os.path.exists(from_checkpoint):
        print(f" Load model từ checkpoint: {from_checkpoint}")
        model = GPT2ForSequenceClassification.from_pretrained(
            from_checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    else:
        print(f" Load model gốc: {MODEL_NAME}")
        model = GPT2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    model.config.pad_token_id = model.config.eos_token_id
    model = model.to(DEVICE)
    return model


# ─── TRAIN 1 EPOCH ───────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="  Training"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        if FP16 and DEVICE == "cuda":
            with autocast("cuda"):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ─── TRAIN TOÀN BỘ ───────────────────────────────────
def train_task(task_name, from_checkpoint=None):
    print(f"\n{'='*50}")
    print(f" Bắt đầu fine-tune: {TASK_NAMES[task_name]}")
    print(f"{'='*50}")

    tokenizer        = get_tokenizer()
    train_dataset, _ = get_dataset(task_name, tokenizer)
    train_loader     = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    model     = load_model(task_name, from_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler    = GradScaler("cuda") if FP16 and DEVICE == "cuda" else None

    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch+1}/{EPOCHS}")
        loss = train_one_epoch(model, train_loader, optimizer, scaler)
        print(f"   Loss: {loss:.4f}")

    save_path = CHECKPOINT[task_name]
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n Đã lưu model tại: {save_path}")

    return model


if __name__ == "__main__":
    print(f"  Device: {DEVICE}")
    train_task("sst2")
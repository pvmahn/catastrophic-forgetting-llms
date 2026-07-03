# train_ewc.py
# Elastic Weight Consolidation (EWC) - phuong phap giam catastrophic forgetting
# Y tuong: them penalty vao loss de bao ve trong so quan trong cua task cu

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import copy
from config import *
from dataset_loader import get_dataset, get_tokenizer, get_num_labels
from train import load_model


# --- EWC CLASS -----------------------------------------------
class EWC:
    def __init__(self, model, dataloader, device):
        self.model   = model
        self.device  = device
        self.params  = {}
        self.fisher  = {}

        for name, param in model.named_parameters():
            self.params[name] = param.data.clone().cpu()

        self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        print("  Tinh Fisher Information Matrix...")

        for name, param in self.model.named_parameters():
            self.fisher[name] = torch.zeros_like(param.data.cpu())

        self.model.eval()
        count = 0

        for batch in tqdm(dataloader, desc="  Fisher"):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
            loss = outputs.loss
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data.clone().cpu() ** 2

            count += 1
            if count >= 50:
                break

        for name in self.fisher:
            self.fisher[name] /= count

        print("  Tinh Fisher xong!")

    def penalty(self, model):
        loss = torch.tensor(0.0)

        for name, param in model.named_parameters():
            if name in self.fisher:
                if self.fisher[name].shape != param.shape:
                    continue
                fisher    = self.fisher[name]
                old_p     = self.params[name]
                param_cpu = param.data.cpu()
                loss     += (fisher * (param_cpu - old_p) ** 2).sum()

        return loss.to(self.device)


# --- TRAIN 1 EPOCH VOI EWC -----------------------------------
def train_one_epoch_ewc(model, dataloader, optimizer,
                        scaler, ewc_list, ewc_lambda):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="  Training EWC"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        if FP16 and DEVICE == "cuda":
            with autocast("cuda"):
                outputs  = model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
                loss     = outputs.loss

                ewc_loss = torch.tensor(0.0).to(DEVICE)
                for ewc in ewc_list:
                    ewc_loss += ewc.penalty(model)

                total_batch_loss = loss + ewc_lambda * ewc_loss

            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs  = model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
            loss     = outputs.loss

            ewc_loss = torch.tensor(0.0).to(DEVICE)
            for ewc in ewc_list:
                ewc_loss += ewc.penalty(model)

            total_batch_loss = loss + ewc_lambda * ewc_loss
            total_batch_loss.backward()
            optimizer.step()

        total_loss += total_batch_loss.item()

    return total_loss / len(dataloader)


# --- TRAIN TOAN BO VOI EWC -----------------------------------
def train_task_ewc(task_name, from_checkpoint=None,
                   ewc_list=[], ewc_lambda=5000):
    print(f"\n{'='*50}")
    print(f"Bat dau EWC fine-tune: {TASK_NAMES[task_name]}")
    print(f"EWC Lambda: {ewc_lambda}")
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
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss = train_one_epoch_ewc(model, train_loader, optimizer,
                                   scaler, ewc_list, ewc_lambda)
        print(f"  Loss (voi EWC penalty): {loss:.4f}")

    save_path = CHECKPOINT_EWC[task_name]
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nDa luu EWC model tai: {save_path}")

    return model, train_loader
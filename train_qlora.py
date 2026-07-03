# train_qlora.py
# QLoRA (Quantized Low-Rank Adaptation) - giam catastrophic forgetting
# Y tuong: Quantize model 4-bit + chi train cac ma tran low-rank nho

import torch
import json
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from config import *
from dataset_loader import get_dataset, get_tokenizer, get_num_labels


# --- CAU HINH QLORA ------------------------------------------
QLORA_CONFIG = {
    "r"           : 16,
    "lora_alpha"  : 32,
    "lora_dropout": 0.1,
    "bias"        : "none",
}

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# --- LOAD MODEL VOI QLORA ------------------------------------
def load_model_qlora(task_name, from_checkpoint=None):
    """
    Load GPT-2 quantize 4-bit va them QLoRA adapter
    - from_checkpoint: load QLoRA checkpoint cu neu co
    - Mac dinh: load GPT-2 goc 4-bit + them QLoRA adapter moi
    """
    num_labels = get_num_labels(task_name)

    if from_checkpoint and os.path.exists(from_checkpoint):
        print(f" Load QLoRA model tu checkpoint: {from_checkpoint}")

        # Đọc num_labels cũ từ config.json
        config_path = os.path.join(from_checkpoint, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                old_config = json.load(f)
            old_num_labels = old_config.get("num_labels", num_labels)
        else:
            old_num_labels = num_labels

        # Load base model 4-bit với num_labels CŨ
        base_model = GPT2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=old_num_labels,
            quantization_config=BNB_CONFIG,
            ignore_mismatched_sizes=True
        )
        base_model.config.pad_token_id = base_model.config.eos_token_id
        base_model = prepare_model_for_kbit_training(base_model)

        try:
            model = PeftModel.from_pretrained(
                base_model,
                from_checkpoint,
                is_trainable=True
            )
            print(f" Load QLoRA adapter thanh cong!")

            # Thay score.weight nếu num_labels khác nhau
            if old_num_labels != num_labels:
                print(f" Thay score: {old_num_labels} → {num_labels} labels")
                model.base_model.model.score = torch.nn.Linear(
                    1024, num_labels, bias=False
                ).to(DEVICE)

        except RuntimeError:
            print(f" ⚠️ Size mismatch → Tao QLoRA adapter moi!")
            base_model2 = GPT2ForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=num_labels,
                quantization_config=BNB_CONFIG,
                ignore_mismatched_sizes=True
            )
            base_model2.config.pad_token_id = base_model2.config.eos_token_id
            base_model2 = prepare_model_for_kbit_training(base_model2)

            lora_config = LoraConfig(
                task_type     = TaskType.SEQ_CLS,
                r             = QLORA_CONFIG["r"],
                lora_alpha    = QLORA_CONFIG["lora_alpha"],
                lora_dropout  = QLORA_CONFIG["lora_dropout"],
                bias          = QLORA_CONFIG["bias"],
                target_modules= ["c_attn", "c_proj"],
            )
            model = get_peft_model(base_model2, lora_config)
            print(f" Da tao QLoRA adapter moi!")

    else:
        print(f" Load GPT-2 4-bit + them QLoRA adapter moi")

        # Load model với quantization 4-bit
        base_model = GPT2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            quantization_config=BNB_CONFIG,
            ignore_mismatched_sizes=True
        )
        base_model.config.pad_token_id = base_model.config.eos_token_id

        # Chuẩn bị model cho k-bit training
        base_model = prepare_model_for_kbit_training(base_model)

        # Cấu hình QLoRA
        lora_config = LoraConfig(
            task_type     = TaskType.SEQ_CLS,
            r             = QLORA_CONFIG["r"],
            lora_alpha    = QLORA_CONFIG["lora_alpha"],
            lora_dropout  = QLORA_CONFIG["lora_dropout"],
            bias          = QLORA_CONFIG["bias"],
            target_modules= ["c_attn", "c_proj"],
        )
        model = get_peft_model(base_model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        print(f" Trainable params: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")
        print(f" Model quantized: 4-bit NF4 ")

    return model


# --- TRAIN 1 EPOCH VOI QLORA ---------------------------------
def train_one_epoch_qlora(model, dataloader, optimizer, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="  Training QLoRA"):
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


# --- TRAIN TOAN BO VOI QLORA ---------------------------------
def train_task_qlora(task_name, from_checkpoint=None):
    """
    Fine-tune model voi QLoRA tren 1 task
    - Quantize model 4-bit → tiết kiệm VRAM
    - Chi train QLoRA adapter (it tham so hon nhieu)
    - Trong so goc GPT-2 bi dong bang (frozen)
    """
    print(f"\n{'='*50}")
    print(f" Bat dau QLoRA fine-tune: {TASK_NAMES[task_name]}")
    print(f" QLoRA rank r = {QLORA_CONFIG['r']}")
    print(f" Quantization: 4-bit NF4")
    print(f"{'='*50}")

    tokenizer        = get_tokenizer()
    train_dataset, _ = get_dataset(task_name, tokenizer)
    train_loader     = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    model = load_model_qlora(task_name, from_checkpoint)

    # Đảm bảo QLoRA params được train
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print(" Không có trainable params → train tất cả!")
        trainable_params = list(model.parameters())

    print(f" Trainable params: {len(trainable_params)} tensors")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    scaler    = GradScaler("cuda") if FP16 and DEVICE == "cuda" else None

    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch+1}/{EPOCHS}")
        loss = train_one_epoch_qlora(model, train_loader, optimizer, scaler)
        print(f"   Loss: {loss:.4f}")

    save_path = CHECKPOINT_QLORA[task_name]
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n Da luu QLoRA model tai: {save_path}")

    return model


if __name__ == "__main__":
    print(f"  Device: {DEVICE}")
    train_task_qlora("sst2")
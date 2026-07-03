from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from config import *

# --- TOKENIZER -----------------------------------------------
def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# --- DATASET CLASS -------------------------------------------
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# --- LOAD TUNG DATASET ---------------------------------------
def load_hellaswag(tokenizer, max_train=MAX_TRAIN_SAMPLES, max_eval=MAX_EVAL_SAMPLES):
    print("Loading HellaSwag...")
    dataset = load_dataset("hellaswag", trust_remote_code=True, cache_dir=DATA_DIR)

    def process(split, max_samples):
        data   = dataset[split].select(range(min(max_samples, len(dataset[split]))))
        texts  = [f"{row['ctx']} {row['endings'][0]}" for row in data]
        labels = [int(row['label']) for row in data]
        enc    = tokenizer(texts, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors=None)
        return TextDataset(enc, labels)

    return process("train", max_train), process("validation", max_eval)


def load_medmcqa(tokenizer, max_train=MAX_TRAIN_SAMPLES, max_eval=MAX_EVAL_SAMPLES):
    print("Loading MedMCQA...")
    dataset = load_dataset("medmcqa", trust_remote_code=True, cache_dir=DATA_DIR)

    def process(split, max_samples):
        data   = dataset[split].select(range(min(max_samples, len(dataset[split]))))
        texts  = [f"Question: {row['question']} Answer: {row['opa']}" for row in data]
        labels = [int(row['cop']) for row in data]
        labels = [max(0, min(3, l - 1)) if l > 0 else 0 for l in labels]
        enc    = tokenizer(texts, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors=None)
        return TextDataset(enc, labels)

    return process("train", max_train), process("validation", max_eval)


def load_sst2(tokenizer, max_train=MAX_TRAIN_SAMPLES, max_eval=MAX_EVAL_SAMPLES):
    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2", trust_remote_code=True, cache_dir=DATA_DIR)

    def process(split, max_samples):
        data   = dataset[split].select(range(min(max_samples, len(dataset[split]))))
        texts  = [row['sentence'] for row in data]
        labels = [int(row['label']) for row in data]
        enc    = tokenizer(texts, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors=None)
        return TextDataset(enc, labels)

    return process("train", max_train), process("validation", max_eval)


# --- HAM CHINH -----------------------------------------------
def get_dataset(task_name, tokenizer):
    if task_name == "hellaswag":
        return load_hellaswag(tokenizer)
    elif task_name == "medmcqa":
        return load_medmcqa(tokenizer)
    elif task_name == "sst2":
        return load_sst2(tokenizer)
    else:
        raise ValueError(f"Task khong hop le: {task_name}")


def get_num_labels(task_name):
    return 2 if task_name == "sst2" else 4


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    for task in TASKS:
        train_ds, eval_ds = get_dataset(task, tokenizer)
        print(f"{task}: train={len(train_ds)}, eval={len(eval_ds)}")
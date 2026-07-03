import gradio as gr
import torch
import torch.nn as nn
import os
import json
import random
import re
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from config import *

# --- AUTO DETECT MOI TRUONG ---
IS_KAGGLE = os.path.exists("/kaggle")

# --- CAU HINH ---
MAX_LENGTH = 128

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

def clean_text(text):
    """Bo cac tag [header] [title] [step] [substeps] cua HellaSwag"""
    text = re.sub(r'\[header\]|\[title\]|\[step\]|\[substeps\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- LOAD EXAMPLES TU DATASET THAT (KEM 4 LUA CHON) ---
def load_random_examples(n=100):
    print("📥 Loading examples từ dataset thật...")
    try:
        hellaswag = load_dataset("hellaswag", split="validation")
        idxs = random.sample(range(len(hellaswag)), n)
        hellaswag_examples = []
        for i in idxs:
            item = hellaswag[i]
            ctx = clean_text(item["ctx"])
            endings = [clean_text(e) for e in item["endings"]]
            hellaswag_examples.append({
                "context": ctx,
                "options": {
                    "A": endings[0],
                    "B": endings[1],
                    "C": endings[2],
                    "D": endings[3],
                },
                "input_text": ctx,  # text dua vao model
            })
        print(f"✅ HellaSwag: {n} câu")
    except Exception as e:
        print(f"⚠️ Không load được HellaSwag: {e}")
        hellaswag_examples = [{
            "context": "She poured the milk into the bowl and picked up a spoon.",
            "options": {"A": "(không có dữ liệu)", "B": "(không có dữ liệu)",
                         "C": "(không có dữ liệu)", "D": "(không có dữ liệu)"},
            "input_text": "She poured the milk into the bowl and picked up a spoon.",
        }]

    try:
        medmcqa = load_dataset("medmcqa", split="validation")
        idxs = random.sample(range(len(medmcqa)), n)
        medmcqa_examples = []
        for i in idxs:
            item = medmcqa[i]
            question = item["question"]
            medmcqa_examples.append({
                "context": question,
                "options": {
                    "A": item["opa"],
                    "B": item["opb"],
                    "C": item["opc"],
                    "D": item["opd"],
                },
                "input_text": f"Question: {question} Answer: {item['opa']}",
            })
        print(f"✅ MedMCQA: {n} câu")
    except Exception as e:
        print(f"⚠️ Không load được MedMCQA: {e}")
        medmcqa_examples = [{
            "context": "Which drug is used as an anticoagulant?",
            "options": {"A": "Warfarin", "B": "Aspirin", "C": "Paracetamol", "D": "Insulin"},
            "input_text": "Question: Which drug is used as an anticoagulant? Answer: Warfarin",
        }]

    return hellaswag_examples, medmcqa_examples

hellaswag_examples, medmcqa_examples = load_random_examples(100)

# --- KET QUA THUC NGHIEM TREN KAGGLE (1000 cau eval) ---
EXPERIMENT_RESULTS = {
    "hellaswag": {
        "task_name": "HellaSwag (Commonsense Reasoning)",
        "num_labels": 4,
        "label_map": {0: "A", 1: "B", 2: "C", 3: "D"},
        "checkpoints": {
            "sequential": CHECKPOINT["sst2"],
            "ewc":        CHECKPOINT_EWC["sst2"],
            "qlora":      CHECKPOINT_QLORA["sst2"],
            "peak":       CHECKPOINT["hellaswag"],
        },
        "before_cf": {"sequential": 30.9, "ewc": 32.7, "qlora": 27.3},
        "after_cf":  {"sequential": 20.6, "ewc": 23.8, "qlora": 24.2},
        "forgetting": {"sequential": 10.3, "ewc": 8.9,  "qlora": 3.1},
        "examples": hellaswag_examples,
    },
    "medmcqa": {
        "task_name": "MedMCQA (Medical QA)",
        "num_labels": 4,
        "label_map": {0: "A", 1: "B", 2: "C", 3: "D"},
        "checkpoints": {
            "sequential": CHECKPOINT["sst2"],
            "ewc":        CHECKPOINT_EWC["sst2"],
            "qlora":      CHECKPOINT_QLORA["sst2"],
            "peak":       CHECKPOINT["medmcqa"],
        },
        "before_cf": {"sequential": 54.3, "ewc": 47.2, "qlora": 59.3},
        "after_cf":  {"sequential": 48.8, "ewc": 46.0, "qlora": 56.2},
        "forgetting": {"sequential": 5.5,  "ewc": 1.2,  "qlora": 3.1},
        "examples": medmcqa_examples,
    },
}

model_cache = {}

def load_model_demo(path, num_labels, is_qlora=False):
    if path in model_cache:
        return model_cache[path]
    if not os.path.exists(path):
        return None, None
    try:
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(path)
        except Exception:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token

        if is_qlora:
            try:
                if IS_KAGGLE:
                    base_model = GPT2ForSequenceClassification.from_pretrained(
                        MODEL_NAME,
                        num_labels=num_labels,
                        quantization_config=BNB_CONFIG,
                        ignore_mismatched_sizes=True
                    )
                else:
                    base_model = GPT2ForSequenceClassification.from_pretrained(
                        MODEL_NAME,
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True
                    )
                base_model.config.pad_token_id = base_model.config.eos_token_id

                model = PeftModel.from_pretrained(
                    base_model,
                    path,
                    is_trainable=False,
                    ignore_mismatched_sizes=True
                )

                # Fix thu cong score.weight tu 2 → 4 lop
                try:
                    old_weight = model.base_model.score.original_module.weight.data
                    if old_weight.shape[0] < num_labels:
                        new_linear = nn.Linear(old_weight.shape[1], num_labels, bias=False)
                        new_linear.weight.data[:old_weight.shape[0]] = old_weight
                        model.base_model.score.original_module = new_linear
                        print(f"✅ Resize score.weight: {old_weight.shape[0]} → {num_labels} lop!")
                except Exception as e:
                    print(f"⚠️ Khong resize duoc: {e}")

                if not IS_KAGGLE:
                    model = model.to(DEVICE)
                print("✅ Load QLoRA adapter thanh cong!")
            except Exception as e:
                print(f"❌ Load QLoRA that bai: {e}")
                return None, None
        else:
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    ckpt_num_labels = json.load(f).get("num_labels", num_labels)
            else:
                ckpt_num_labels = num_labels
            model = GPT2ForSequenceClassification.from_pretrained(
                path,
                num_labels=ckpt_num_labels,
                ignore_mismatched_sizes=True
            )
            model.config.pad_token_id = model.config.eos_token_id
            model = model.to(DEVICE)

        model.eval()
        model_cache[path] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None


def predict_single(model, tokenizer, text):
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE)
        )
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        pred  = int(probs.argmax())
        conf  = float(probs[pred])
    return pred, conf, probs


def make_prob_bar(probs, label_map, highlight_idx):
    lines = []
    for i, prob in enumerate(probs):
        bar  = "█" * int(prob * 20)
        mark = " ← DỰ ĐOÁN" if i == highlight_idx else ""
        lines.append(f"  {label_map[i]}: {bar} {prob*100:.1f}%{mark}")
    return "\n".join(lines)


def format_options(options):
    return "\n".join([f"  {k}. {v}" for k, v in options.items()])


def analyze(task_key, example_idx):
    cfg        = EXPERIMENT_RESULTS[task_key]
    examples   = cfg["examples"]
    example    = examples[example_idx]
    input_text = example["input_text"]
    options    = example["options"]
    context    = example["context"]

    ckpts      = cfg["checkpoints"]
    label_map  = cfg["label_map"]
    num_labels = cfg["num_labels"]

    m_seq,   t_seq   = load_model_demo(ckpts["sequential"], num_labels)
    m_ewc,   t_ewc   = load_model_demo(ckpts["ewc"],        num_labels)
    m_qlora, t_qlora = load_model_demo(ckpts["qlora"],      num_labels, is_qlora=True)
    m_peak,  t_peak  = load_model_demo(ckpts["peak"],       num_labels)

    missing = []
    if not m_seq:   missing.append("Sequential")
    if not m_ewc:   missing.append("EWC")
    if not m_qlora: missing.append("QLoRA")
    if not m_peak:  missing.append("Baseline")
    if missing:
        return [f"❌ Thiếu checkpoint: {', '.join(missing)}", "", "", "", "", ""]

    pred_peak,  conf_peak,  probs_peak  = predict_single(m_peak,  t_peak,  input_text)
    pred_seq,   conf_seq,   probs_seq   = predict_single(m_seq,   t_seq,   input_text)
    pred_ewc,   conf_ewc,   probs_ewc   = predict_single(m_ewc,   t_ewc,   input_text)
    pred_qlora, conf_qlora, probs_qlora = predict_single(m_qlora, t_qlora, input_text)

    peak_label  = label_map[pred_peak]
    seq_label   = label_map[pred_seq]
    ewc_label   = label_map[pred_ewc]
    qlora_label = label_map[pred_qlora]

    seq_cf   = pred_seq   != pred_peak
    ewc_cf   = pred_ewc   != pred_peak
    qlora_cf = pred_qlora != pred_peak

    def cf_icon(is_cf):
        return "🔴 Bị CF (đổi dự đoán)" if is_cf else "🟢 Không bị CF (giữ dự đoán)"

    qlora_note = "" if IS_KAGGLE else " *(⚠️ local: không có 4-bit)*"

    context_box = f"""📖 Ngữ cảnh / Câu hỏi:
{context}

🔤 4 lựa chọn:
{format_options(options)}"""

    summary = f"""## 🔍 Kết quả minh họa trên câu này

> **Mục đích:** Xem model có đổi dự đoán sau khi học thêm task mới không?
> Nếu đổi → CF xảy ra! Nếu giữ nguyên → phương pháp bảo vệ tốt!

| Phương pháp | Dự đoán | Confidence | CF xảy ra không? |
|-------------|---------|-----------|-----------------|
| 🟢 Baseline (ngay sau Task 1) | **{peak_label}** | {conf_peak*100:.1f}% | ✅ Chưa học thêm task mới |
| 🔴 Sequential (sau Task 3) | **{seq_label}** | {conf_seq*100:.1f}% | {cf_icon(seq_cf)} |
| 🔵 EWC (sau Task 3) | **{ewc_label}** | {conf_ewc*100:.1f}% | {cf_icon(ewc_cf)} |
| 🟣 QLoRA (sau Task 3){qlora_note} | **{qlora_label}** | {conf_qlora*100:.1f}% | {cf_icon(qlora_cf)} |

> ⚠️ **Lưu ý:** Kết quả trên **1 câu** không đại diện cho toàn bộ!
> Xem **Forgetting Score** ở trên (đo trên 1000 câu) để đánh giá chính xác.
{'' if IS_KAGGLE else '> 🟣 **QLoRA local:** Không có 4-bit quantization (Windows). Forgetting **3.1%** được đo chính xác trên Kaggle Linux!'}
"""

    prob_peak_str  = make_prob_bar(probs_peak,  label_map, pred_peak)
    prob_seq_str   = make_prob_bar(probs_seq,   label_map, pred_seq)
    prob_ewc_str   = make_prob_bar(probs_ewc,   label_map, pred_ewc)
    prob_qlora_str = make_prob_bar(probs_qlora, label_map, pred_qlora)

    return [context_box, summary, prob_peak_str, prob_seq_str, prob_ewc_str, prob_qlora_str]


def next_example(task_key):
    """Tra ve index ngau nhien + hien thi ngu canh/4 lua chon"""
    examples = EXPERIMENT_RESULTS[task_key]["examples"]
    idx = random.randint(0, len(examples) - 1)
    ex = examples[idx]
    context_box = f"""📖 Ngữ cảnh / Câu hỏi:
{ex['context']}

🔤 4 lựa chọn:
{format_options(ex['options'])}"""
    return idx, context_box


def get_forgetting_table(task_key):
    cfg    = EXPERIMENT_RESULTS[task_key]
    before = cfg["before_cf"]
    after  = cfg["after_cf"]
    frg    = cfg["forgetting"]
    return f"""📊 KẾT QUẢ THỰC NGHIỆM — {cfg['task_name']}
Đo trên 1000 câu eval | GPU T4 16GB | Kaggle
═══════════════════════════════════════════════════

  Phương pháp      Trước CF → Sau CF   Forgetting
  ─────────────────────────────────────────────────
  🔴 Sequential    {before['sequential']}%    → {after['sequential']}%     -{frg['sequential']}% ❌ quên nhiều nhất
  🔵 EWC (λ=5000)  {before['ewc']}%    → {after['ewc']}%     -{frg['ewc']}%  ✅
  🟣 QLoRA (4-bit) {before['qlora']}%    → {after['qlora']}%     -{frg['qlora']}%  ✅ quên ít nhất
  ─────────────────────────────────────────────────

  Forgetting Score = Accuracy(Trước CF) - Accuracy(Sau CF)
  Càng thấp → càng ít quên → phương pháp càng tốt!"""


# --- GIAO DIEN ---
with gr.Blocks(title="Catastrophic Forgetting Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧠 Catastrophic Forgetting in Continual Fine-tuning of LLMs

    **Vấn đề:** Khi fine-tune LLM trên nhiều task liên tiếp,
    model có thể **quên kiến thức cũ** — gọi là *Catastrophic Forgetting (CF)*.

    **Thực nghiệm:** Fine-tune **GPT-2 Medium (345M params)** trên 3 task theo thứ tự:
    `Task 1: HellaSwag` (suy luận) → `Task 2: MedMCQA` (y tế) → `Task 3: SST-2` (cảm xúc)

    **So sánh 3 phương pháp giảm CF:** Sequential | EWC | QLoRA
    """)

    gr.Markdown("---")

    # PHẦN 1: KẾT QUẢ THỰC NGHIỆM CỐ ĐỊNH
    gr.Markdown("""## 📊 Kết quả thực nghiệm
*Số liệu cố định — đo trên 1000 câu eval thực tế trên Kaggle*""")

    with gr.Row():
        gr.Textbox(
            value=get_forgetting_table("hellaswag"),
            label="📌 Task 1: HellaSwag — Commonsense Reasoning",
            lines=11, interactive=False
        )
        gr.Textbox(
            value=get_forgetting_table("medmcqa"),
            label="📌 Task 2: MedMCQA — Medical QA",
            lines=11, interactive=False
        )

    gr.Markdown("""
    > **Giải thích:** Sau khi train qua cả 3 task, model bị quên kiến thức Task 1 và Task 2.
    > Forgetting Score đo mức độ quên đó. **QLoRA quên ít nhất** vì đóng băng GPT-2 gốc!
    """)

    gr.Markdown("---")

    # PHẦN 2: DEMO TƯƠNG TÁC
    gr.Markdown("""## 🔬 Demo tương tác
*Minh họa CF trực quan — xem model có đổi dự đoán sau khi học thêm task không*""")

    with gr.Row():
        with gr.Column(scale=1):

            task_radio = gr.Radio(
                choices=["hellaswag", "medmcqa"],
                value="hellaswag",
                label="Chọn task để kiểm tra",
                info="hellaswag: câu suy luận | medmcqa: câu y tế"
            )

            context_display = gr.Textbox(
                label="Ngữ cảnh & 4 lựa chọn",
                lines=8, interactive=False,
                value=f"""📖 Ngữ cảnh / Câu hỏi:
{hellaswag_examples[0]['context']}

🔤 4 lựa chọn:
{format_options(hellaswag_examples[0]['options'])}"""
            )
            example_idx_state = gr.State(value=0)

            with gr.Row():
                example_btn = gr.Button("📝 Câu ví dụ ngẫu nhiên", variant="secondary")
                run_btn     = gr.Button("🔍 Kiểm tra CF", variant="primary", scale=2)

            gr.Markdown("""
            **3 Phương pháp fine-tuning:**
            | | Phương pháp | Cơ chế giảm CF |
            |--|--|--|
            | 🔴 | Sequential | Không bảo vệ |
            | 🔵 | EWC | Fisher Matrix |
            | 🟣 | QLoRA | Frozen GPT-2 + Adapter |

            🟢 **Baseline** = GPT-2 ngay sau Task 1, **chưa bị CF**
            """)

        with gr.Column(scale=2):
            output_summary = gr.Markdown(
                value="*Nhấn **Kiểm tra CF** để xem model có quên không...*"
            )

    gr.Markdown("---")

    # PHẦN 3: PHÂN BỐ XÁC SUẤT
    gr.Markdown("""## 🎯 Phân bố xác suất dự đoán
*Xem phương pháp nào giữ được dự đoán giống Baseline nhất*""")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🟢 Baseline\n*Trước khi bị CF*")
            output_peak = gr.Textbox(lines=6, interactive=False, show_label=False)

        with gr.Column():
            gr.Markdown("### 🔴 Sequential\n*Bị CF nhiều nhất*")
            output_seq = gr.Textbox(lines=6, interactive=False, show_label=False)

        with gr.Column():
            gr.Markdown("### 🔵 EWC\n*Giảm CF bằng Fisher*")
            output_ewc = gr.Textbox(lines=6, interactive=False, show_label=False)

        with gr.Column():
            gr.Markdown(
                "### 🟣 QLoRA\n*Giảm CF tốt nhất*\n*(⚠️ Local: không có 4-bit)*"
                if not IS_KAGGLE else
                "### 🟣 QLoRA\n*Giảm CF tốt nhất*"
            )
            output_qlora = gr.Textbox(lines=6, interactive=False, show_label=False)

    # Events
    def on_task_change(task_key):
        idx, context_box = next_example(task_key)
        return idx, context_box

    task_radio.change(
        fn=on_task_change,
        inputs=[task_radio],
        outputs=[example_idx_state, context_display]
    )
    example_btn.click(
        fn=on_task_change,
        inputs=[task_radio],
        outputs=[example_idx_state, context_display]
    )
    run_btn.click(
        fn=analyze,
        inputs=[task_radio, example_idx_state],
        outputs=[context_display, output_summary, output_peak, output_seq, output_ewc, output_qlora]
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  CATASTROPHIC FORGETTING DEMO")
    print("  GPT-2 Medium | Sequential vs EWC vs QLoRA")
    print("=" * 55)
    print(f"\n  Device    : {DEVICE.upper()}")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Moi truong: {'Kaggle' if IS_KAGGLE else 'Local'}")
    print("\n  Kiem tra checkpoint...")

    all_paths = set()
    for cfg in EXPERIMENT_RESULTS.values():
        for p in cfg["checkpoints"].values():
            all_paths.add(p)

    for path in sorted(all_paths):
        status = "✅" if os.path.exists(path) else "❌ Chua co"
        print(f"  {status} {path}")

    print("\n  🚀 Khoi dong: http://localhost:7860")
    print("=" * 55)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=IS_KAGGLE,
        show_error=True
    )
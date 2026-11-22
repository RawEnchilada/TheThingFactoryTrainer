"""Writes notebooks/train_colab.ipynb. Edit the cells here, then run:
    python notebooks/make_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# ForgeTrainer: fine-tune Gemma 4 E4B on Colab

Trains the forge model for The Thing Factory on a free Colab T4 (15 GB).

**Before you start (on your PC):**
1. Collect jobs with `THING_FACTORY_TRAINING=true`, drop them in `training_things/`.
2. `python -m forge_trainer build` writes `data/train.jsonl` and `data/eval.jsonl`.
3. Copy both into Google Drive at **`MyDrive/ForgeTrainer/data/`**.

**Here:** Runtime > Change runtime type > **T4 GPU**, then Runtime > Run all.

Checkpoints go to `MyDrive/ForgeTrainer/output/`, so if Colab disconnects (sessions end after a few hours),
just run everything again: training resumes from the last checkpoint.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
%%capture
!pip install unsloth
""")

md("## Settings")

code("""
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
ROOT = Path("/content/drive/MyDrive/ForgeTrainer")
DATA = ROOT / "data"
OUT = ROOT / "output"

MODEL = "unsloth/gemma-4-E4B-it"
MAX_LENGTH = 6144      # longer samples are dropped (the script prompts are the long ones); 8192 may not fit a T4
LORA_R = 16
EPOCHS = 2
LEARNING_RATE = 2e-4
GRAD_ACCUM = 8
SAVE_EVERY = 25        # steps between checkpoints on Drive
QUANT = "q4_k_m"       # GGUF quantization for Ollama

OUT.mkdir(parents=True, exist_ok=True)
print(sorted(p.name for p in DATA.glob("*")) if DATA.exists() else f"missing {DATA}: copy train.jsonl / eval.jsonl there")
""")

md("## Data")

code("""
import json
from collections import Counter

def rows(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_rows = rows(DATA / "train.jsonl")
eval_rows = rows(DATA / "eval.jsonl")
assert train_rows, f"no training data in {DATA}"
print(f"{len(train_rows)} train / {len(eval_rows)} eval samples")
print("per step:", dict(Counter(r["kind"] for r in train_rows)))
""")

md("## Model (4-bit base + LoRA)")

code("""
from unsloth import FastModel
import torch

model, processor = FastModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_LENGTH, load_in_4bit=True, full_finetuning=False)
model = FastModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0, bias="none", random_state=3407)
tokenizer = getattr(processor, "tokenizer", processor)
BF16 = torch.cuda.is_bf16_supported()
print("bf16" if BF16 else "fp16 (no bf16 on this GPU)")
""")

md("""
## Tokenize

Each sample is rendered with the model's own chat template. The loss only covers the answer: every token of the
system + user prompt is masked. This doesn't depend on knowing Gemma's turn markers, because the prompt is rendered on
its own (with the generation prompt) and must be an exact prefix of the full conversation.
""")

code("""
def as_parts(messages):
    # Processors want list content; plain tokenizers want strings.
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]

def render(messages, generation_prompt):
    for template_owner, msgs in ((processor, as_parts(messages)), (tokenizer, messages)):
        try:
            return template_owner.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation_prompt)
        except Exception:
            continue
    raise RuntimeError("no chat template")

def encode(row):
    prompt_text = render(row["prompt"], True)
    full_text = render(row["prompt"] + row["completion"], False)
    if not full_text.startswith(prompt_text):
        return None
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) > MAX_LENGTH:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

def encode_all(source):
    kept, dropped = [], Counter()
    for r in source:
        e = encode(r)
        if e is None:
            dropped[r["kind"]] += 1
        else:
            kept.append(e)
    return kept, dropped

train_enc, train_dropped = encode_all(train_rows)
eval_enc, _ = encode_all(eval_rows)
print(f"kept {len(train_enc)} train / {len(eval_enc)} eval")
if train_dropped:
    print(f"dropped (longer than {MAX_LENGTH} tokens or template mismatch):", dict(train_dropped))
assert train_enc, "nothing left to train on"

# Sanity check: what the loss sees for the first sample.
first = train_enc[0]
print("--- trained part of sample 0 ---")
print(tokenizer.decode([t for t, l in zip(first["input_ids"], first["labels"]) if l != -100])[:800])
""")

md("## Train (resumes from the last checkpoint on Drive)")

code("""
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

ckpt_dir = OUT / "checkpoints"
resume = any(ckpt_dir.glob("checkpoint-*"))
print("resuming" if resume else "starting fresh")

args = SFTConfig(
    output_dir=str(ckpt_dir),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="steps",
    save_steps=SAVE_EVERY,
    save_total_limit=2,
    eval_strategy="steps" if eval_enc else "no",
    eval_steps=SAVE_EVERY,
    max_length=MAX_LENGTH,
    bf16=BF16,
    fp16=not BF16,
    optim="adamw_8bit",
    report_to="none",
    seed=3407,
    dataset_kwargs={"skip_prepare_dataset": True},  # already tokenized and masked above
    remove_unused_columns=False,
)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=Dataset.from_list(train_enc),
    eval_dataset=Dataset.from_list(eval_enc) if eval_enc else None,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
)
trainer.train(resume_from_checkpoint=resume or None)
""")

md("## Save the adapter")

code("""
adapter = OUT / "adapter"
model.save_pretrained(str(adapter))
processor.save_pretrained(str(adapter))
(adapter / "forge_trainer.json").write_text(json.dumps(
    {"base_model": MODEL, "backend": "unsloth", "train_samples": len(train_enc), "eval_samples": len(eval_enc)}, indent=1))
print("saved", adapter)
""")

md("## Try it on a held-out prompt")

code("""
FastModel.for_inference(model)
sample = (eval_rows or train_rows)[0]
text = render(sample["prompt"], True)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
out = model.generate(**inputs, max_new_tokens=600, do_sample=False)
print("step:", sample["kind"])
print("--- model ---")
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print("--- teacher ---")
print(sample["completion"][0]["content"][:1500])
""")

md("""
## Export a GGUF for Ollama

Merging needs a lot of system RAM. If this cell runs out of memory, skip it: download `output/adapter/` from Drive
and export on your PC instead (`python -m forge_trainer export --backend hf --llama-cpp <llama.cpp> --ollama-name forge-gemma`).
""")

code("""
gguf_dir = OUT / "gguf"
model.save_pretrained_gguf(str(gguf_dir), processor, quantization_method=QUANT)
print(sorted(p.name for p in gguf_dir.glob("*.gguf")))
""")

md("""
## On your PC

Download the `.gguf` from `MyDrive/ForgeTrainer/output/gguf/`, then:

```sh
ollama pull gemma4:e4b      # only for its chat template
python -m forge_trainer export --gguf path/to/model.gguf --ollama-name forge-gemma
```

In the game: Options > Forge > **Ollama (local)**, model `forge-gemma`.
""")

nb = {
    "cells": [dict(c, source=[line + "\n" for line in c["source"].split("\n")][:-1] + [c["source"].split("\n")[-1]]) for c in CELLS],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).with_name("train_colab.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)

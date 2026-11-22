"""LoRA fine-tune of Gemma 4 E4B on the forge stages.

Reads data/train.jsonl (+ data/eval.jsonl), written by `build`. One model
learns every stage: the system prompt already tells it which stage it is in,
exactly as in the game.

Backends:
- unsloth (default when it is installed): FastModel, faster, less memory.
- hf: plain transformers + peft + trl, for machines Unsloth doesn't support.

The heavy imports happen inside train() so the rest of the package (ingest,
watch, build) works without torch.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "unsloth/gemma-4-E4B-it"


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    model: str = DEFAULT_MODEL
    backend: str = "auto"  # auto | unsloth | hf
    max_length: int = 8192
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8
    resume: bool = False


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pick_backend(name: str) -> str:
    if name != "auto":
        return name
    try:
        import unsloth  # noqa: F401
        return "unsloth"
    except Exception:  # not installed, or no supported GPU
        return "hf"


def _load(cfg: TrainConfig, backend: str):
    if backend == "unsloth":
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=cfg.model, max_seq_length=cfg.max_length,
            load_in_4bit=cfg.load_in_4bit, full_finetuning=False)
        model = FastModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0, bias="none",
            random_state=3407)
        return model, tokenizer

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"torch_dtype": torch.bfloat16}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
    except ValueError:
        # E4B is published as an image-text-to-text model; text training still works.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(cfg.model, **kwargs)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    return model, tokenizer


def _fits(tokenizer, row: dict, max_length: int) -> bool:
    ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
    return len(ids) <= max_length


def train(cfg: TrainConfig) -> int:
    train_rows = _rows(cfg.data_dir / "train.jsonl")
    eval_rows = _rows(cfg.data_dir / "eval.jsonl")
    if not train_rows:
        print("no training data: drop job folders into training_things/ and run `build`", file=sys.stderr)
        return 1
    backend = _pick_backend(cfg.backend)
    print(f"backend {backend}, model {cfg.model}, {len(train_rows)} train / {len(eval_rows)} eval samples", flush=True)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = _load(cfg, backend)
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)  # multimodal processors wrap one
    kept = [r for r in train_rows if _fits(text_tokenizer, r, cfg.max_length)]
    if len(kept) < len(train_rows):
        print(f"dropped {len(train_rows) - len(kept)} samples longer than {cfg.max_length} tokens", flush=True)
    eval_kept = [r for r in eval_rows if _fits(text_tokenizer, r, cfg.max_length)]

    def as_dataset(rows):
        return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

    args = SFTConfig(
        output_dir=str(cfg.output_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_kept else "no",
        max_length=cfg.max_length,
        completion_only_loss=True,  # learn the answers, not the prompts
        bf16=True,
        report_to="none",
        seed=3407,
    )
    trainer = SFTTrainer(
        model=model, processing_class=text_tokenizer, args=args,
        train_dataset=as_dataset(kept), eval_dataset=as_dataset(eval_kept) if eval_kept else None)
    trainer.train(resume_from_checkpoint=cfg.resume or None)
    adapter = cfg.output_dir / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    (adapter / "forge_trainer.json").write_text(json.dumps(
        {"base_model": cfg.model, "backend": backend, "train_samples": len(kept), "eval_samples": len(eval_kept)}, indent=1))
    print(f"adapter saved to {adapter}", flush=True)
    return 0

"""python -m forge_trainer <command>  (see README.md)"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import dataset
from .store import Store

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forge_trainer", description="Train a small model on The Thing Factory's forge pipeline.")
    p.add_argument("--things", type=Path, default=ROOT / "training_things", help="where job folders are dropped")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="ingested samples and datasets")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="process new or changed job folders once")
    w = sub.add_parser("watch", help="ingest job folders as soon as they are dropped in")
    w.add_argument("--interval", type=float, default=1.0)
    w.add_argument("--train-every", type=int, default=0, help="start training after this many new jobs (0 = never)")
    sub.add_parser("stats", help="what is in the data")
    b = sub.add_parser("build", help="write data/train.jsonl and data/eval.jsonl")
    b.add_argument("--eval-percent", type=int, default=10)
    b.add_argument("--kinds", default="", help="only these stage kinds, comma separated (outline,model,sound,texture,sprite,script,repair)")

    t = sub.add_parser("train", help="build, then LoRA fine-tune")
    t.add_argument("--output", type=Path, default=ROOT / "output")
    t.add_argument("--model", default=None)
    t.add_argument("--backend", choices=["auto", "unsloth", "hf"], default="auto")
    t.add_argument("--max-length", type=int, default=8192)
    t.add_argument("--no-4bit", action="store_true", help="bf16 LoRA (more VRAM) instead of QLoRA")
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--eval-percent", type=int, default=10)

    e = sub.add_parser("export", help="adapter -> GGUF -> Ollama model")
    e.add_argument("--output", type=Path, default=ROOT / "output")
    e.add_argument("--quant", default="q4_k_m")
    e.add_argument("--llama-cpp", type=Path, default=None, help="llama.cpp checkout (only without Unsloth)")
    e.add_argument("--template-from", default="gemma4:e4b", help="Ollama model whose chat template to copy")
    e.add_argument("--ollama-name", default="", help="create this Ollama model (e.g. forge-gemma)")
    e.add_argument("--backend", choices=["auto", "unsloth", "hf"], default="auto")
    e.add_argument("--gguf", type=Path, default=None, help="use this ready GGUF (e.g. from the Colab notebook) instead of converting")

    args = p.parse_args(argv)
    store = Store(args.things, args.data)

    if args.command == "ingest":
        for key, entry in store.sync():
            print(_line(key, entry))
        print(dataset.stats(store.all_samples(), store.state))
        return 0
    if args.command == "watch":
        from .watch import watch
        store.sync()
        watch(store, args.interval, args.train_every)
        return 0
    if args.command == "stats":
        store.sync()
        print(dataset.stats(store.all_samples(), store.state))
        return 0
    if args.command == "build":
        store.sync()
        kinds = {k.strip() for k in args.kinds.split(",") if k.strip()} or None
        print(dataset.build(store.all_samples(), args.data, args.eval_percent, kinds))
        return 0
    if args.command == "train":
        store.sync()
        info = dataset.build(store.all_samples(), args.data, args.eval_percent)
        print(info)
        from .train import DEFAULT_MODEL, TrainConfig, train
        cfg = TrainConfig(
            data_dir=args.data, output_dir=args.output, model=args.model or DEFAULT_MODEL, backend=args.backend,
            max_length=args.max_length, load_in_4bit=not args.no_4bit, epochs=args.epochs, learning_rate=args.lr,
            lora_r=args.lora_r, lora_alpha=args.lora_r * 2, grad_accum=args.grad_accum, resume=args.resume)
        return train(cfg)
    if args.command == "export":
        from .export import export
        return export(args.output / "adapter", args.output / "export", args.quant, args.llama_cpp,
                      args.template_from, args.ollama_name, args.backend, args.gguf)
    return 2


def _line(key: str, entry: dict) -> str:
    if entry.get("removed"):
        return f"- {key}: removed"
    if entry["samples"]:
        return f"+ {key}: {entry['samples']} samples ({entry['status']})"
    return f"  {key}: no samples ({entry['skipped']})"


if __name__ == "__main__":
    sys.exit(main())

"""From a trained adapter to an Ollama model the game can use.

1. GGUF: with Unsloth, `save_pretrained_gguf` merges and quantizes in one go.
   Without it the adapter is merged with peft into output/merged/ and
   llama.cpp's convert_hf_to_gguf.py (--llama-cpp <checkout>) makes the GGUF.
2. Modelfile: `FROM ./<file>.gguf`, plus TEMPLATE / PARAMETER lines copied from
   an Ollama model with the same chat format (--template-from, default
   gemma4:e4b), so the chat template matches what the model was trained on.
3. `ollama create <name>` when --ollama-name is given.

In the game: Options > Forge > Ollama (local), model = that name.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _gguf_unsloth(adapter: Path, out: Path, quant: str) -> Path:
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(model_name=str(adapter), load_in_4bit=False)
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=quant)
    found = sorted(out.glob("*.gguf"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise RuntimeError(f"no .gguf written to {out}")
    return found[-1]


def _gguf_llama_cpp(adapter: Path, out: Path, quant: str, llama_cpp: Path | None) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = json.loads((adapter / "forge_trainer.json").read_text())["base_model"]
    merged = out / "merged"
    try:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged))
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(merged))
    if llama_cpp is None:
        raise RuntimeError(f"merged model in {merged}; pass --llama-cpp <llama.cpp checkout> to make the GGUF")
    f16 = out / "model-f16.gguf"
    subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"], check=True)
    quantize = shutil.which("llama-quantize") or str(llama_cpp / "build/bin/llama-quantize")
    final = out / f"model-{quant}.gguf"
    subprocess.run([quantize, str(f16), str(final), quant.upper()], check=True)
    return final


def _template_lines(source: str) -> list[str]:
    """TEMPLATE / PARAMETER / SYSTEM lines of an installed Ollama model, or []."""
    if not shutil.which("ollama"):
        return []
    try:
        text = subprocess.run(["ollama", "show", source, "--modelfile"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    keep, lines, inside = [], text.splitlines(), False
    for line in lines:
        if line.startswith("TEMPLATE"):
            inside = '"""' in line and line.count('"""') == 1
            keep.append(line)
        elif inside:
            keep.append(line)
            if '"""' in line:
                inside = False
        elif line.startswith("PARAMETER"):
            keep.append(line)
    return keep


def export(adapter: Path, out: Path, quant: str = "q4_k_m", llama_cpp: Path | None = None,
           template_from: str = "gemma4:e4b", ollama_name: str = "", backend: str = "auto") -> int:
    out.mkdir(parents=True, exist_ok=True)
    use_unsloth = backend == "unsloth"
    if backend == "auto":
        try:
            import unsloth  # noqa: F401
            use_unsloth = True
        except Exception:
            use_unsloth = False
    gguf = _gguf_unsloth(adapter, out, quant) if use_unsloth else _gguf_llama_cpp(adapter, out, quant, llama_cpp)
    print(f"gguf: {gguf}", flush=True)

    template = _template_lines(template_from)
    if not template:
        print(f"warning: could not read the chat template of {template_from!r} from Ollama "
              f"(`ollama pull {template_from}` first); the Modelfile relies on the GGUF's own template", file=sys.stderr)
    modelfile = out / "Modelfile"
    modelfile.write_text("\n".join([f"FROM ./{gguf.relative_to(out).as_posix()}", *template]) + "\n", encoding="utf-8")
    print(f"Modelfile: {modelfile}", flush=True)

    if ollama_name:
        subprocess.run(["ollama", "create", ollama_name, "-f", str(modelfile)], check=True, cwd=str(out))
        print(f"ollama model {ollama_name!r} created: in the game pick Options > Forge > Ollama (local) and set the model to it", flush=True)
    return 0

# ForgeTrainer

Fine-tunes **Gemma 4 E4B** (`unsloth/gemma-4-E4B-it`) to be The Thing Factory's forge model. It trains every step of the game's **staged pipeline** (`forge/pipeline/staged_builder.gd` in TheSandbox): outline, model, sound, texture, sprite, script and repair. One model learns all of them, because each step's system prompt already says which step it is, exactly as in the game.

## How the data gets here

1. Start the game with **`THING_FACTORY_TRAINING=true`** in its environment (e.g. `THING_FACTORY_TRAINING=true ./TheThingFactory`; on Steam: launch options `THING_FACTORY_TRAINING=true %command%`). Without it the pipeline writes no samples and the game clears old jobs on startup.
2. Build things in the game with a strong teacher **through the pipeline** (not the agent): Options > Forge, e.g. Claude Code or Claude API, and `"builder": "pipeline"` in `runner_config.json` for Claude Code (its default is the agent builder, which writes no samples).
3. Each accepted step is saved as `stages/<stage>.sample.json` in the job folder: the exact system + user prompt the model was asked and the accepted answer.
4. Copy job folders from `~/.local/share/godot/app_userdata/The Thing Factory/forge/jobs/` (Windows: `%APPDATA%\Godot\app_userdata\The Thing Factory\forge\jobs\`) into `training_things/`. You can copy one job, a few, or the whole `jobs/` folder. With training on, jobs survive restarts; without it the game deletes them at startup.

What counts:

| Job result | What is kept |
|---|---|
| `ok` | every accepted step. If the game test needed repairs, the script step's target is the **last repair's code** (the one that passed); earlier failed repairs are dropped; the last repair is kept as a `repair` sample |
| `unsafe` | only an outline that refused (`refuse: true`), which teaches refusals |
| `failed`, still running, no samples | nothing (`stats` lists why) |

## Use

Ingest, watch and build need only Python 3.11+ (standard library):

```sh
python -m forge_trainer watch                    # picks up dropped job folders within ~1-2 s
python -m forge_trainer watch --train-every 50   # and starts training after every 50 new jobs
python -m forge_trainer ingest                   # one pass instead of watching
python -m forge_trainer stats                    # samples per step, skipped jobs and why
python -m forge_trainer build                    # data/train.jsonl + data/eval.jsonl
```

`training_things/` is the source of truth: a changed folder is ingested again, and a removed one loses its samples (`data/samples/<folder>.jsonl`, state in `data/ingested.json`). A folder is read once it has stopped changing for one poll, so copying a big folder is fine.

Training and export need the `train` extra (and Unsloth if your GPU supports it):

```sh
uv venv --python 3.12 && uv pip install -e ".[train,unsloth]"
python -m forge_trainer train                    # QLoRA, 2 epochs -> output/adapter/
python -m forge_trainer export --ollama-name forge-gemma
```

`train` builds the dataset first, then trains a LoRA (r 16, all attention + MLP projections, loss on the answers only via TRL's prompt/completion format). Samples longer than `--max-length` (8192) tokens are dropped and counted. The eval split (10% of **jobs**, stable by hash) is used once there are 10+ jobs. Options: `--no-4bit` (bf16 LoRA, more VRAM), `--epochs`, `--lr`, `--lora-r`, `--backend hf` (plain transformers + peft without Unsloth), `--resume`.

`export` makes a GGUF (`--quant q4_k_m`), writes a Modelfile that copies the chat template from an installed Ollama model with the same format (`--template-from gemma4:e4b`, pull it first) and runs `ollama create`. Without Unsloth it merges with peft and needs a llama.cpp checkout (`--llama-cpp <dir>`). In the game: Options > Forge > **Ollama (local)**, model = `forge-gemma`.

## Hardware

Gemma 4 E4B QLoRA needs roughly 10-17 GB of VRAM depending on sequence length; the script step's prompts are long (several thousand tokens). An NVIDIA GPU with Unsloth is the easy path; a rented 24-48 GB GPU or Colab works. AMD (ROCm) can use `--backend hf`, but bitsandbytes 4-bit on ROCm and consumer RDNA2 cards are not reliably supported.

## Tests

```sh
python -m unittest discover tests
```

`tests/fixtures/job_ok` is a real job from the game's replay test.

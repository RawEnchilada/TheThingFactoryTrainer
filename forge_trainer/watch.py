"""Watches training_things/ and ingests job folders as soon as they land.

Polling (stdlib only, works the same on every OS and on network drives): a
new or changed folder is ingested once it has stayed the same for one poll,
so a folder still being copied is not read half-written. With
`train_every`, a training run starts in the background after that many new
jobs with samples (one run at a time).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .store import Store


def _describe(key: str, entry: dict) -> str:
    if entry.get("removed"):
        return f"- {key}: removed"
    if entry["samples"]:
        return f"+ {key}: {entry['samples']} samples ({entry['status']})"
    return f"  {key}: no samples ({entry['skipped']})"


def watch(store: Store, interval: float = 1.0, train_every: int = 0, train_args: list[str] | None = None) -> None:
    print(f"watching {store.things_dir} (Ctrl+C to stop)", flush=True)
    seen: dict[str, str] = {}  # key -> fingerprint from the previous poll
    new_jobs = 0
    training: subprocess.Popen | None = None
    try:
        while True:
            pending = store.pending()
            for key, (job_dir, fp) in pending.items():
                if seen.get(key) == fp:  # unchanged for a whole poll: done copying
                    entry = store.ingest(key, job_dir, fp)
                    print(_describe(key, entry), flush=True)
                    if entry["samples"]:
                        new_jobs += 1
            seen = {k: fp for k, (_, fp) in pending.items()}
            for key in store.prune():
                print(_describe(key, {"removed": True}), flush=True)

            if training is not None and training.poll() is not None:
                print(f"training finished (exit {training.returncode})", flush=True)
                training = None
            if train_every and new_jobs >= train_every and training is None:
                print(f"{new_jobs} new jobs: starting training", flush=True)
                training = subprocess.Popen([sys.executable, "-m", "forge_trainer", "train", *(train_args or [])])
                new_jobs = 0
            time.sleep(interval)
    except KeyboardInterrupt:
        print("stopped", flush=True)

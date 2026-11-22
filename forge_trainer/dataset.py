"""From ingested samples to the files the trainer reads.

data/train.jsonl and data/eval.jsonl use TRL's conversational prompt /
completion format, so the loss only covers the answer:

    {"prompt": [system, user], "completion": [assistant], "kind": "outline", "job": "..."}

The split is per job (all of a job's stages land on the same side), decided
by a hash of the job id so it stays stable as more jobs arrive.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

MIN_JOBS_FOR_EVAL = 10


def _in_eval(job: str, eval_percent: int) -> bool:
    return int(hashlib.sha1(job.encode()).hexdigest(), 16) % 100 < eval_percent


def to_record(sample: dict) -> dict:
    system, user, answer = sample["messages"]
    return {"prompt": [system, user], "completion": [answer], "kind": sample["kind"], "job": sample["job"]}


def build(samples: list[dict], out_dir: Path, eval_percent: int = 10, kinds: set[str] | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if kinds:
        samples = [s for s in samples if s["kind"] in kinds]
    jobs = {s["job"] for s in samples}
    use_eval = eval_percent > 0 and len(jobs) >= MIN_JOBS_FOR_EVAL
    train, evals = [], []
    for s in samples:
        (evals if use_eval and _in_eval(s["job"], eval_percent) else train).append(to_record(s))
    for name, rows in (("train", train), ("eval", evals)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"train": len(train), "eval": len(evals), "jobs": len(jobs), "eval_used": use_eval}


def stats(samples: list[dict], state: dict) -> str:
    lines = []
    statuses = Counter(e.get("status", "") for e in state.values())
    with_samples = sum(1 for e in state.values() if e.get("samples"))
    lines.append(f"job folders: {len(state)} ({with_samples} with samples)  statuses: {dict(statuses)}")
    skipped = Counter(e["skipped"] for e in state.values() if not e.get("samples") and e.get("skipped"))
    for reason, n in skipped.most_common():
        lines.append(f"  skipped {n}: {reason}")
    lines.append(f"samples: {len(samples)}")
    for kind, n in sorted(Counter(s["kind"] for s in samples).items()):
        lines.append(f"  {kind:8} {n}")
    fixed = sum(1 for s in samples if s.get("fixed_by_repair"))
    if fixed:
        lines.append(f"  (script targets taken from a passing repair: {fixed})")
    teachers = Counter(f"{s['teacher']['provider']}/{s['teacher']['model']}" for s in samples)
    lines.append("teachers: " + ", ".join(f"{t} ({n})" for t, n in teachers.most_common()))
    return "\n".join(lines)

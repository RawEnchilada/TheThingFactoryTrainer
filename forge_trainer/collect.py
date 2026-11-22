"""Turn one forge job folder into training samples.

A job folder is what The Thing Factory's forge leaves in
`user://forge/jobs/<job>/`. The staged pipeline writes, for every stage whose
answer it accepted, `stages/<stage>.sample.json`:

    {"stage", "system", "user", "schema", "answer", "attempts", "model", "provider"}

`user` is the first-attempt input (without the "your previous answer had
problems" notes), so a sample is exactly what the model is asked in the game
and the answer it should give.

Rules:
- Only jobs with result.json status "ok" give samples, since a failed job's answers
  are not known to work.
- Refusals ("unsafe") give their outline sample only when it refused
  (`refuse: true`): that teaches the model to refuse.
- The script stage's own answer may have been fixed later by repair calls
  after the game tested it. The last repair is the code that passed, so it
  becomes the script stage's target. Earlier repairs failed in the game and
  are dropped. The last repair is also kept as a "repair" sample.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JobResult:
    job_id: str
    status: str  # result.json status, or "" when missing
    samples: list[dict] = field(default_factory=list)
    skipped: str = ""  # why this job gave no samples


def stage_kind(stage: str) -> str:
    """'sound:hum.wav' -> 'sound', 'repair_2' -> 'repair'."""
    return re.sub(r"_\d+$", "", stage.split(":")[0])


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _answer_text(answer) -> str:
    """What the model has to output: JSON for structured stages, raw text for code."""
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, ensure_ascii=False)


def _repair_number(stage: str) -> int:
    m = re.search(r"_(\d+)$", stage)
    return int(m.group(1)) if m else 0


def collect_job(job_dir: Path) -> JobResult:
    job_dir = Path(job_dir)
    job = _read_json(job_dir / "job.json")
    result = _read_json(job_dir / "result.json")
    job_id = (job or {}).get("id") or job_dir.name
    out = JobResult(job_id=str(job_id), status=str((result or {}).get("status", "")))
    if not isinstance(job, dict):
        out.skipped = "no job.json"
        return out
    if not isinstance(result, dict):
        out.skipped = "no result.json (still running?)"
        return out
    raw = []
    for path in sorted((job_dir / "stages").glob("*.sample.json")):
        data = _read_json(path)
        if isinstance(data, dict) and {"stage", "system", "user", "answer"} <= data.keys():
            raw.append(data)
    if not raw:
        out.skipped = "no stages/*.sample.json (agent builder, or a game version before samples)"
        return out

    meta = {"job": out.job_id, "prompt": job.get("prompt", ""), "edit": bool(job.get("edit", False))}
    if out.status == "unsafe":
        for s in raw:
            if stage_kind(s["stage"]) == "outline" and isinstance(s["answer"], dict) and s["answer"].get("refuse"):
                out.samples.append(_sample(s, meta))
        if not out.samples:
            out.skipped = "refused, but not by the outline (the scan or the agent refused)"
        return out
    if out.status != "ok":
        out.skipped = f"job status {out.status!r}"
        return out

    repairs = sorted((s for s in raw if stage_kind(s["stage"]) == "repair"), key=lambda s: _repair_number(s["stage"]))
    final_repair = repairs[-1] if repairs else None
    for s in raw:
        kind = stage_kind(s["stage"])
        if kind == "repair":
            if s is final_repair:
                out.samples.append(_sample(s, meta))
            continue
        if kind == "script" and final_repair is not None:
            # Same question, but the answer that actually worked in the game.
            s = dict(s, answer=final_repair["answer"])
            out.samples.append(_sample(s, meta, fixed_by_repair=True))
            continue
        out.samples.append(_sample(s, meta))
    return out


def _sample(s: dict, meta: dict, fixed_by_repair: bool = False) -> dict:
    return {
        "stage": s["stage"],
        "kind": stage_kind(s["stage"]),
        "messages": [
            {"role": "system", "content": s["system"]},
            {"role": "user", "content": s["user"]},
            {"role": "assistant", "content": _answer_text(s["answer"])},
        ],
        "schema": s.get("schema") or {},
        "teacher": {"provider": s.get("provider", ""), "model": s.get("model", ""), "attempts": s.get("attempts", 1)},
        "fixed_by_repair": fixed_by_repair,
        **meta,
    }

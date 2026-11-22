"""Keeps data/ in step with training_things/.

Every job folder found under training_things/ (any depth up to MAX_DEPTH, so
dropping a whole `jobs/` folder works too) is turned into
data/samples/<key>.jsonl. data/ingested.json remembers each folder's
fingerprint so only new or changed folders are processed again; folders
that were removed lose their samples. training_things/ is the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .collect import collect_job

MAX_DEPTH = 3


class Store:
    def __init__(self, things_dir: Path, data_dir: Path):
        self.things_dir = Path(things_dir)
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"
        self.state_path = self.data_dir / "ingested.json"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.state: dict = self._load_state()

    def _load_state(self) -> dict:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def find_jobs(self) -> list[Path]:
        jobs = []

        def walk(d: Path, depth: int) -> None:
            if (d / "job.json").is_file():
                jobs.append(d)
                return
            if depth >= MAX_DEPTH:
                return
            try:
                children = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
            except OSError:
                return
            for child in children:
                walk(child, depth + 1)

        if self.things_dir.is_dir():
            walk(self.things_dir, 0)
        return jobs

    def key_of(self, job_dir: Path) -> str:
        rel = job_dir.relative_to(self.things_dir).as_posix()
        return rel.replace("/", "__")

    @staticmethod
    def fingerprint(job_dir: Path) -> str:
        """Changes whenever a file that matters for training changes."""
        parts = []
        files = [job_dir / "job.json", job_dir / "result.json", *sorted((job_dir / "stages").glob("*.sample.json"))]
        for f in files:
            try:
                st = f.stat()
                parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                parts.append(f"{f.name}:-")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()

    def pending(self) -> dict[str, tuple[Path, str]]:
        """key -> (folder, fingerprint) for folders that are new or changed."""
        out = {}
        for job_dir in self.find_jobs():
            key = self.key_of(job_dir)
            fp = self.fingerprint(job_dir)
            if self.state.get(key, {}).get("fingerprint") != fp:
                out[key] = (job_dir, fp)
        return out

    def ingest(self, key: str, job_dir: Path, fp: str) -> dict:
        result = collect_job(job_dir)
        path = self.samples_dir / f"{key}.jsonl"
        if result.samples:
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for s in result.samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        elif path.exists():
            path.unlink()
        entry = {
            "fingerprint": fp,
            "job": result.job_id,
            "status": result.status,
            "samples": len(result.samples),
            "skipped": result.skipped,
        }
        self.state[key] = entry
        self._save_state()
        return entry

    def prune(self) -> list[str]:
        """Forget folders that are gone from training_things/."""
        present = {self.key_of(d) for d in self.find_jobs()}
        gone = [k for k in self.state if k not in present]
        for key in gone:
            (self.samples_dir / f"{key}.jsonl").unlink(missing_ok=True)
            del self.state[key]
        if gone:
            self._save_state()
        return gone

    def sync(self) -> list[tuple[str, dict]]:
        """One full pass: ingest everything new or changed, drop what's gone."""
        done = []
        for key, (job_dir, fp) in self.pending().items():
            done.append((key, self.ingest(key, job_dir, fp)))
        for key in self.prune():
            done.append((key, {"removed": True}))
        return done

    def all_samples(self) -> list[dict]:
        samples = []
        for path in sorted(self.samples_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                samples.extend(json.loads(line) for line in f if line.strip())
        return samples

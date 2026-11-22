"""python -m unittest discover tests  (stdlib only, no torch needed)"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from forge_trainer import dataset
from forge_trainer.collect import collect_job, stage_kind
from forge_trainer.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "job_ok"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def sample(stage: str, answer, system: str = "sys", user: str = "usr") -> dict:
    return {"stage": stage, "system": system, "user": user, "schema": {}, "answer": answer,
            "attempts": 1, "model": "m", "provider": "p"}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def job(self, name: str, status: str | None, samples: list[dict], job_id: str | None = None) -> Path:
        d = self.tmp / name
        write_json(d / "job.json", {"id": job_id or name, "prompt": "a lamp", "thing_id": "lamp_1", "edit": False})
        if status is not None:
            write_json(d / "result.json", {"status": status, "reason": ""})
        for s in samples:
            write_json(d / "stages" / (s["stage"].replace(":", "_") + ".sample.json"), s)
        return d


class CollectTest(Base):
    def test_stage_kind(self) -> None:
        self.assertEqual(stage_kind("sound:hum.wav"), "sound")
        self.assertEqual(stage_kind("repair_2"), "repair")
        self.assertEqual(stage_kind("outline"), "outline")

    def test_real_job(self) -> None:
        r = collect_job(FIXTURE)
        self.assertEqual(r.status, "ok")
        self.assertEqual(sorted(s["kind"] for s in r.samples), ["model", "outline", "script", "sound", "texture"])
        outline = next(s for s in r.samples if s["kind"] == "outline")
        roles = [m["role"] for m in outline["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertIsInstance(json.loads(outline["messages"][2]["content"]), dict, "structured answers are JSON text")
        script = next(s for s in r.samples if s["kind"] == "script")
        self.assertIn("func ", script["messages"][2]["content"], "code answers stay raw text")
        self.assertEqual(outline["prompt"], "a glowing orb")

    def test_failed_and_running_jobs_give_nothing(self) -> None:
        self.assertEqual(collect_job(self.job("f", "failed", [sample("outline", {"refuse": False})])).samples, [])
        running = collect_job(self.job("r", None, [sample("outline", {"refuse": False})]))
        self.assertEqual(running.samples, [])
        self.assertIn("result.json", running.skipped)

    def test_refusal_keeps_only_the_refusing_outline(self) -> None:
        r = collect_job(self.job("u", "unsafe", [sample("outline", {"refuse": True, "refuse_reason": "x"})]))
        self.assertEqual([s["kind"] for s in r.samples], ["outline"])
        r = collect_job(self.job("u2", "unsafe", [sample("outline", {"refuse": False}), sample("script", "code")]))
        self.assertEqual(r.samples, [], "the scan refused it, not the model")

    def test_script_target_comes_from_the_last_repair(self) -> None:
        r = collect_job(self.job("rep", "ok", [
            sample("script", "broken", user="write it"),
            sample("repair_1", "still broken", user="fix 1"),
            sample("repair_2", "works", user="fix 2"),
        ]))
        by_stage = {s["stage"]: s for s in r.samples}
        self.assertEqual(set(by_stage), {"script", "repair_2"}, "failed repairs are dropped")
        self.assertEqual(by_stage["script"]["messages"][1]["content"], "write it", "same question")
        self.assertEqual(by_stage["script"]["messages"][2]["content"], "works", "answer that passed the game")
        self.assertTrue(by_stage["script"]["fixed_by_repair"])

    def test_no_samples(self) -> None:
        r = collect_job(self.job("old", "ok", []))
        self.assertEqual(r.samples, [])
        self.assertIn("sample", r.skipped)


class StoreTest(Base):
    def setUp(self) -> None:
        super().setUp()
        self.things = self.tmp / "training_things"
        self.store = Store(self.things, self.tmp / "data")

    def test_add_change_remove(self) -> None:
        shutil.copytree(FIXTURE, self.things / "job_ok")
        done = self.store.sync()
        self.assertEqual(len(done), 1)
        self.assertEqual(len(self.store.all_samples()), 5)
        self.assertEqual(self.store.sync(), [], "nothing changed, nothing redone")

        # A later result (the same folder copied again with a failed status) is picked up.
        time.sleep(0.01)
        write_json(self.things / "job_ok" / "result.json", {"status": "failed"})
        self.store.sync()
        self.assertEqual(self.store.all_samples(), [])

        shutil.rmtree(self.things / "job_ok")
        self.store.sync()
        self.assertEqual(self.store.state, {})

    def test_nested_jobs_folder(self) -> None:
        shutil.copytree(FIXTURE, self.things / "jobs" / "a")
        shutil.copytree(FIXTURE, self.things / "jobs" / "b")
        self.store.sync()
        self.assertEqual(set(self.store.state), {"jobs__a", "jobs__b"})

    def test_state_survives_a_restart(self) -> None:
        shutil.copytree(FIXTURE, self.things / "job_ok")
        self.store.sync()
        again = Store(self.things, self.tmp / "data")
        self.assertEqual(again.pending(), {})


class DatasetTest(Base):
    def test_build_prompt_completion(self) -> None:
        samples = collect_job(FIXTURE).samples
        info = dataset.build(samples, self.tmp / "out")
        self.assertEqual(info["train"], 5)
        self.assertFalse(info["eval_used"], "too few jobs for an eval split")
        row = json.loads((self.tmp / "out" / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual([m["role"] for m in row["prompt"]], ["system", "user"])
        self.assertEqual([m["role"] for m in row["completion"]], ["assistant"])

    def test_split_keeps_jobs_together(self) -> None:
        samples = []
        for i in range(40):
            for kind in ("outline", "model"):
                samples.append({"kind": kind, "job": f"job{i}", "messages": [
                    {"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]})
        info = dataset.build(samples, self.tmp / "out", eval_percent=25)
        self.assertTrue(info["eval_used"])
        self.assertGreater(info["eval"], 0)
        train_jobs = {json.loads(l)["job"] for l in (self.tmp / "out" / "train.jsonl").read_text().splitlines()}
        eval_jobs = {json.loads(l)["job"] for l in (self.tmp / "out" / "eval.jsonl").read_text().splitlines()}
        self.assertEqual(train_jobs & eval_jobs, set())

    def test_kind_filter(self) -> None:
        info = dataset.build(collect_job(FIXTURE).samples, self.tmp / "out", kinds={"outline"})
        self.assertEqual(info["train"], 1)


if __name__ == "__main__":
    unittest.main()

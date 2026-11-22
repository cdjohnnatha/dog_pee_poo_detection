"""tools/shadow_report.py on a synthetic ring: a later quiet frame gives the temporal pair."""

import importlib.util
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ellie.ring import ReferenceRing

F22 = Path(__file__).parent.parent / "fixtures" / "pad_cycle_1122"
spec = importlib.util.spec_from_file_location(
    "shadow_report", Path(__file__).parent.parent.parent / "tools" / "shadow_report.py"
)
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def test_report_finds_a_later_ring_frame_and_pairs_pad_changes(tmp_path):
    t0 = datetime(2026, 11, 22, 9, 0, tzinfo=UTC)
    used = (F22 / "112434_used.jpg").read_bytes()
    vdir = tmp_path / "visits" / "2026" / "11" / "22" / "v_20261122_090000"
    vdir.mkdir(parents=True)
    (vdir / "before.jpg").write_bytes(used)
    (vdir / "after.jpg").write_bytes(used)
    os.utime(vdir / "after.jpg", (t0.timestamp(), t0.timestamp()))
    ring = ReferenceRing(tmp_path / "ring")
    for minute in (1, 3, 5):
        ring.add(used, t0 + timedelta(minutes=minute), "minute", {"busy": False})
    ring.shadow_visit(
        {
            "visit_id": "v_20261122_090000",
            "started_at": "2026-11-22T08:59:00Z",
            "pair": {"before": "before.jpg", "after": "after.jpg", "after_mtime": t0.timestamp()},
            "gemini": {"pee": False},
            "app_outcome": "no_pee",
            "local": {"answer": "uncertain"},
        }
    )
    ring.shadow_pad_change({"at": "2026-11-22T09:10:00+00:00", "source": "visit"})
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "labels.jsonl").write_text(
        json.dumps({"visit_id": "v_20261122_090000", "label": "no_pee"}) + "\n"
    )
    rep = sr.report(tmp_path, t0 + timedelta(hours=1))
    (v,) = rep["visits"]
    assert v["human"] == "no_pee" and v["alternative"]["min_after"] == 3.0
    assert v["alternative"]["local"]["answer"] == "no_pee"
    assert rep["pad_changes"][0]["cycle_confirmed_at"] is None
    assert "recovered by a later ring frame: 1" in sr.summary(rep)

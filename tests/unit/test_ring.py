"""Reference ring (shadow, ADR-0026): storage, pruning, never raising."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ellie.ring import ReferenceRing, local_stain, reduce
from helpers import make_jpeg

F22 = Path(__file__).parent.parent / "fixtures" / "pad_cycle_1122"
T0 = datetime(2026, 11, 22, 10, 0, tzinfo=UTC)


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_a_frame_is_kept_reduced_with_its_time_and_measures(tmp_path):
    ring = ReferenceRing(tmp_path)
    crop = (F22 / "112434_used.jpg").read_bytes()
    meta = ring.add(crop, T0, "minute", {"busy": False, "visit": None})
    (row,) = lines(tmp_path / "index.jsonl")
    assert row["file"] == "20261122T100000Z_minute.jpg" and row["at"] == T0.isoformat()
    assert (tmp_path / row["file"]).stat().st_size < len(crop) / 3
    assert set(row["features"]) >= {"ir", "tex", "clip", "sharp", "lum"}
    assert row["state"] == "PAD_IN_USE" and meta == row


def test_old_frames_are_pruned_and_the_index_follows(tmp_path):
    ring = ReferenceRing(tmp_path, keep_s=3600)
    jpeg = make_jpeg((120, 120, 120))
    for minute in range(0, 130, 10):
        ring.add(jpeg, T0 + timedelta(minutes=minute), "minute", {})
    stats = ring.prune(T0 + timedelta(minutes=130))
    files = sorted(p.name for p in tmp_path.glob("*Z_*.jpg"))
    assert files[0] >= (T0 + timedelta(minutes=70)).strftime("%Y%m%dT%H%M%SZ")
    assert {r["file"] for r in lines(tmp_path / "index.jsonl")} == set(files)
    assert stats["deleted"] > 0 and stats["kept"] == len(files)


def test_the_count_cap_holds(tmp_path):
    ring = ReferenceRing(tmp_path, max_files=5)
    for second in range(12):
        ring.add(make_jpeg((100, 100, 100)), T0 + timedelta(seconds=second * 60), "minute", {})
    ring.prune(T0 + timedelta(minutes=12))
    assert len(list(tmp_path.glob("*Z_*.jpg"))) == 5


def test_a_bad_frame_or_an_unwritable_folder_never_raises(tmp_path):
    assert ReferenceRing(tmp_path).add(b"not a jpeg", T0, "minute", {}) is None
    blocked = tmp_path / "file"
    blocked.write_text("x")
    assert ReferenceRing(blocked / "ring").add(make_jpeg((1, 2, 3)), T0, "minute", {}) is None


def test_follow_up_frames_only_during_a_pad_cycle(tmp_path):
    ring = ReferenceRing(tmp_path)
    ring.add((F22 / "112434_used.jpg").read_bytes(), T0, "minute", {})
    assert not ring.wants_followup(T0)
    ring.add((F22 / "112435_empty_base.jpg").read_bytes(), T0 + timedelta(seconds=1), "busy", {})
    assert ring.wants_followup(T0 + timedelta(seconds=2))
    assert not ring.wants_followup(T0 + timedelta(hours=1))  # a cycle that never ends stops asking


def test_reduce_keeps_the_aspect_ratio():
    import io

    from PIL import Image

    small = Image.open(io.BytesIO(reduce((F22 / "112434_used.jpg").read_bytes())))
    assert small.width == 640


def test_local_stain_answers_uncertain_when_the_floor_changed():
    used, empty = (
        (F22 / "112434_used.jpg").read_bytes(),
        (F22 / "112435_empty_base.jpg").read_bytes(),
    )
    assert local_stain(used, used)["answer"] == "no_pee"
    assert local_stain(used, empty)["answer"] == "uncertain"


def test_app_pad_changes_are_logged_next_to_the_cycle(tmp_path):
    ring = ReferenceRing(tmp_path / "ring")
    ring.shadow_pad_change({"at": "2026-11-22T11:24:35Z", "source": "visit"})
    (row,) = lines(tmp_path / "ring" / "app_pad_changes.jsonl")
    assert row["source"] == "visit"

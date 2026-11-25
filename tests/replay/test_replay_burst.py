"""A visit with the shadow bursts: frames during and after, the summary, the dataset copy."""

import json

import pytest
from test_replay_visit import EVENT, send

from ellie.ring import ReferenceRing
from helpers import FakeClock, build_rig, make_jpeg


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rig(tmp_path, clock):
    rig = build_rig(tmp_path, clock, ring=ReferenceRing(tmp_path / "ring"))
    rig.reference.update(make_jpeg((200, 200, 200)))
    yield rig
    rig.store.close()


async def advance(rig, seconds):
    """Like the app: one tick a second, and the spawned captures run between ticks."""
    for _ in range(int(seconds)):
        rig.clock.advance(1)
        await rig.service.tick()
        await rig.service.wait_idle()  # the frame grabs (a thread) finish within the second


async def visit_with_bursts(rig, stay=12):
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/dog", 1)
    await rig.service.on_message("frigate/events", json.dumps(EVENT).encode())
    await advance(rig, stay)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)
    await advance(rig, 120)


async def test_frames_during_and_after_and_a_summary(rig, tmp_path):
    await visit_with_bursts(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    d = rig.evidence.visit_dir(visit["id"]) / "burst"
    during = sorted(p.name for p in d.glob("d_*.jpg"))
    after = sorted(p.name for p in d.glob("a_*.jpg"))
    summary = json.loads((d / "burst.json").read_text())
    assert 3 <= len(during) <= 5 and summary["during_frames"] == len(during)
    assert summary["after_frames"] == 9  # a_005 .. a_085, one every 10 s
    best = (summary["best"] or {}).get("file")
    assert after == ([best] if best else [])  # only the best after frame is kept
    lines = [
        json.loads(x) for x in (tmp_path / "ring" / "shadow_visits.jsonl").read_text().splitlines()
    ]
    assert any(x.get("kind") == "burst" and x["visit_id"] == visit["id"] for x in lines)
    assert visit["final_outcome"] == "pee"  # the decision is untouched


async def test_the_posture_frames_go_to_the_dataset_with_the_label(rig):
    await visit_with_bursts(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    await rig.service.on_command(
        "label", json.dumps({"visit_id": visit["id"], "label": "pee"}).encode()
    )
    await rig.service.wait_idle()
    dest = rig.evidence.root / "dataset" / "pee" / visit["id"] / "burst"
    assert list(dest.glob("d_*.jpg")) and (dest / "burst.json").exists()
    assert not list(dest.glob("a_*.jpg"))  # only her posture, not the after frames


async def test_no_ring_no_bursts(tmp_path, clock):
    rig = build_rig(tmp_path, clock)
    rig.reference.update(make_jpeg((200, 200, 200)))
    await visit_with_bursts(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert not (rig.evidence.visit_dir(visit["id"]) / "burst").exists()

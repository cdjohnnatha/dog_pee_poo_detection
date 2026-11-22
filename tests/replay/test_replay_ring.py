"""The reference ring is shadow only: it keeps frames and logs, and changes nothing else."""

import json

import pytest
from test_replay_visit import one_visit

from ellie.ring import ReferenceRing
from helpers import FakeClock, build_rig, make_jpeg


@pytest.fixture
def clock():
    return FakeClock()


def make(tmp_path, clock, ring):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rig = build_rig(tmp_path, clock, **({"ring": ring} if ring else {}))
    rig.reference.update(make_jpeg((200, 200, 200)))
    return rig


async def test_a_visit_with_the_ring_counts_and_publishes_exactly_as_without(tmp_path, clock):
    plain = make(tmp_path / "a", clock, None)
    await one_visit(plain)
    ring_dir = tmp_path / "b" / "ring"
    shadow = make(tmp_path / "b", FakeClock(), ReferenceRing(ring_dir))
    await one_visit(shadow)
    await shadow.service.wait_idle()
    strip = lambda rig: [(t, b) for t, b, _ in rig.published if t != "ellie/state"]
    assert [t for t, _ in strip(plain)] == [t for t, _ in strip(shadow)]
    (a,) = plain.store.visits_started_between("2026", "2028")
    (b,) = shadow.store.visits_started_between("2026", "2028")
    assert (a["auto_outcome"], a["final_outcome"]) == (b["auto_outcome"], b["final_outcome"])
    (record,) = [json.loads(x) for x in (ring_dir / "shadow_visits.jsonl").read_text().splitlines()]
    assert record["visit_id"] == b["id"] and record["gemini"]["pee"] is True
    assert record["local"]["answer"] in ("pee", "no_pee", "uncertain")


async def test_the_minute_reference_also_goes_to_the_ring(tmp_path, clock):
    ring_dir = tmp_path / "ring"
    rig = make(tmp_path, clock, ReferenceRing(ring_dir))
    for _ in range(15):
        clock.advance(1)
        await rig.service.tick()
    await rig.service.minute_tasks()
    await rig.service.wait_idle()
    (row,) = [json.loads(x) for x in (ring_dir / "index.jsonl").read_text().splitlines()]
    assert row["source"] == "minute" and row["context"]["busy"] is False


async def test_a_failing_camera_for_the_ring_does_not_disturb_the_app(tmp_path, clock):
    from ellie.frames import FrameError

    calls = {"n": 0}

    async def broken():
        calls["n"] += 1
        raise FrameError("go2rtc 500")

    rig = make(tmp_path, clock, ReferenceRing(tmp_path / "ring"))
    rig.service._fetch_frame = broken
    await rig.service.on_message("frigate/tray_area/person", b"1")  # busy: ring wants a frame
    await rig.service.minute_tasks()
    await rig.service.wait_idle()
    assert calls["n"] >= 1 and not list((tmp_path / "ring").glob("*.jpg"))

"""Experimental pad cycle (shadow, ADR-0026) on real frames.

Regression 2026-11-22: the app recorded a pad change at 11:24:35 on a frame of the EMPTY base (no pad,
no arcs); the new pad was put in at ~11:29. An empty or removed tray must never confirm a change.
"""

from datetime import datetime, timedelta
from pathlib import Path

from ellie.tray_state import PadCycle, features

F22 = Path(__file__).parent.parent / "fixtures" / "pad_cycle_1122"
F09 = Path(__file__).parent.parent / "fixtures" / "pad_cycle_1109"


def at(day, hms):
    return datetime.fromisoformat(f"2026-11-{day}T{hms}+00:00")


def run(cycle, frames):
    return [cycle.observe(t, (folder / name).read_bytes()) for t, folder, name in frames]


SEQ_1122 = [
    (at(22, "11:24:34"), F22, "112434_used.jpg"),
    (at(22, "11:24:35"), F22, "112435_empty_base.jpg"),
    (at(22, "11:25:36"), F22, "112536_removed.jpg"),
    (at(22, "11:27:20"), F22, "112720_cleaning.jpg"),
    (at(22, "11:29:00"), F22, "112900_new_pad.jpg"),
    (at(22, "11:59:00"), F22, "115900_new_pad_stable.jpg"),
]


def test_the_empty_base_of_2026_11_22_is_not_a_pad_change():
    out = run(PadCycle(), SEQ_1122)
    assert out[1]["state"] == "TRAY_EMPTY"  # 11:24:35, where the app recorded the change
    assert all(o.get("event") != "CONFIRMED" for o in out)


def test_2026_11_22_confirms_only_when_the_new_pad_is_seen_steady_twice():
    stable = (F22 / "115900_new_pad_stable.jpg").read_bytes()
    cycle = PadCycle()
    run(cycle, SEQ_1122)
    # what the ring adds: the same steady view one and three minutes later
    pending = cycle.observe(at(22, "12:00:00"), stable)
    done = cycle.observe(at(22, "12:02:00"), stable)
    assert pending["state"] == "PAD_REPLACED_PENDING" and pending["event"] == "PENDING"
    assert done["event"] == "CONFIRMED"
    start, end = done["replaced_between"]
    assert start <= at(22, "11:24:34").isoformat() and end >= at(22, "11:29:00").isoformat()


def test_2026_11_09_real_change_is_confirmed_with_an_honest_interval():
    """The owner marked it at 13:30; the frames show it happened between 13:06 and 13:10."""
    out = run(PadCycle(), [
        (at("09", "12:59:00"), F09, "1259_used.jpg"),
        (at("09", "13:17:17"), F09, "1317_new_pad.jpg"),  # 18 min gap: not steady yet
        (at("09", "13:19:00"), F09, "1319_new_pad.jpg"),
        (at("09", "13:28:26"), F09, "1328_new_pad.jpg"),
    ])  # fmt: skip
    assert out[1]["state"] == "UNKNOWN"
    assert out[2]["event"] == "PENDING" and out[3]["event"] == "CONFIRMED"
    start, end = out[3]["replaced_between"]
    assert start <= at("09", "13:06:00").isoformat() and end >= at("09", "13:10:00").isoformat()


def test_the_same_pad_put_back_is_not_a_change():
    used = (F22 / "112434_used.jpg").read_bytes()
    cycle = PadCycle()
    cycle.observe(at(22, "10:00:00"), used)
    cycle.observe(at(22, "10:01:00"), (F22 / "112435_empty_base.jpg").read_bytes())
    moving = cycle.observe(at(22, "10:05:00"), used)  # first frame back: the scene just changed
    back = cycle.observe(at(22, "10:06:00"), used)
    later = cycle.observe(at(22, "10:07:00"), used)
    assert moving["state"] == "CLEANING"
    assert back["event"] == "SAME_PAD_BACK" and back["state"] == "PAD_IN_USE"
    assert later["state"] == "PAD_IN_USE" and "event" not in later


def test_a_busy_frame_never_starts_or_confirms_a_change():
    cycle = PadCycle()
    cycle.observe(at(22, "11:24:34"), (F22 / "112434_used.jpg").read_bytes())
    new = (F22 / "115900_new_pad_stable.jpg").read_bytes()
    for minute in range(5):
        out = cycle.observe(at(22, "11:30:00") + timedelta(minutes=minute), new, busy=True)
        assert out["state"] in ("UNKNOWN", "CLEANING")


def test_texture_tells_the_empty_base_from_pads():
    tex = {p.name: features(p.read_bytes())["tex"] for p in F22.glob("*.jpg")}
    assert tex["112435_empty_base.jpg"] < 2.2
    assert min(v for k, v in tex.items() if k != "112435_empty_base.jpg") > 5

"""T152 (2026-11-25 09:59): the owner turned the tray 180°; the moved stains were counted as a pee
by the promotion check. A real person (big box) at the tray is never Ellie: no promotion."""

import json

import pytest
from test_replay_pad import person_at_the_tray, presences
from test_replay_visit import build

from helpers import FakeClock, make_jpeg, make_pad_result, make_result

OWNER = {"id": "1795600756.0-owner1", "camera": "ellie", "label": "person", "box": [3, 29, 305, 386],
         "area": 107800, "current_zones": ["tray_area"], "entered_zones": ["tray_area"]}  # fmt: skip


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rig(tmp_path, clock):
    rig = build(tmp_path, clock)
    rig.reference.update(make_jpeg((200, 200, 200)))
    rig.analyser.script_pad(make_pad_result(pad_replaced=False, pad_replaced_confidence=0.95))
    rig.analyser.script(make_result(pee=True, pee_confidence=0.95))  # "nova mancha de urina"
    yield rig
    rig.store.close()


async def owner_event(rig, kind):
    after = OWNER | ({"end_time": 1.0} if kind == "end" else {})
    await rig.service.on_message(
        "frigate/events", json.dumps({"type": kind, "after": after}).encode()
    )


async def test_the_owner_turning_the_tray_is_not_a_pee(rig):
    await owner_event(rig, "new")
    await person_at_the_tray(rig)
    await owner_event(rig, "end")
    for _ in range(60):  # the held event kept the presence going; now the area is clear
        rig.clock.advance(1)
        await rig.service.tick()
    await rig.service.wait_idle()
    (presence,) = presences(rig)
    assert presence["final_outcome"] == "not_replaced"
    assert rig.store.visits_started_between("2026", "2028") == []  # no promoted pee
    assert presence["promotion_pending"] is False


async def test_a_small_person_alone_is_still_checked(rig):
    """Without a big person event the old check stays (Ellie seen as a person, ADR-0011)."""
    await person_at_the_tray(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["final_outcome"] == "pee"

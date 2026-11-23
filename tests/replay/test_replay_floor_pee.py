"""A pee on the floor (ADR-0029 in spec 003): banner state, owner mark, cleaning, pad stats."""

import json

import pytest
from test_replay_visit import one_visit, topics

from helpers import FakeClock, build_rig, make_jpeg, make_result


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rig(tmp_path, clock):
    rig = build_rig(tmp_path, clock)
    rig.reference.update(make_jpeg((200, 200, 200)))
    yield rig
    rig.store.close()


def floor(rig):
    return topics(rig, "ellie/state")[-1]["floor_pee"]


async def cmd(rig, name, **data):
    await rig.service.on_command(name, json.dumps(data).encode())
    await rig.service.wait_idle()


async def test_the_analyser_seeing_a_pee_off_the_pad_turns_the_banner_on(rig):
    rig.analyser.script(make_result(pee=True, pee_where="off_pad"))
    await one_visit(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    state = floor(rig)
    assert state["on"] and state["visit_id"] == visit["id"] and state["by"] == "analyser"
    await cmd(rig, "floor_clear")
    assert floor(rig)["on"] is False
    v = rig.store.get_visit(visit["id"])
    assert v["floor_cleared_by"] == "owner" and v["final_outcome"] == "pee"  # still counted


async def test_the_owner_marks_the_last_pee_as_on_the_floor(rig):
    rig.analyser.script(make_result(pee=True, pee_where="on_pad"))
    await one_visit(rig)
    assert floor(rig)["on"] is False
    await cmd(rig, "floor_pee")
    (visit,) = rig.store.visits_started_between("2026", "2028")
    v = rig.store.get_visit(visit["id"])
    assert (v["pee_where"], v["pee_where_by"]) == ("both", "owner")
    assert floor(rig)["on"] and floor(rig)["by"] == "owner"
    assert topics(rig, "ellie/pad/full") == []  # the owner already knows: no notice


async def test_a_reanalysis_keeps_the_owners_floor_mark(rig):
    rig.analyser.script(
        make_result(pee=True, pee_where="on_pad"), make_result(pee=True, pee_where="on_pad")
    )
    await one_visit(rig)
    await cmd(rig, "floor_pee")
    (visit,) = rig.store.visits_started_between("2026", "2028")
    await cmd(rig, "reanalyse", visit_id=visit["id"])
    v = rig.store.get_visit(visit["id"])
    assert (v["pee_where"], v["pee_where_by"]) == ("both", "owner")


async def test_with_no_recent_pee_the_mark_adds_the_missed_pee(rig):
    await cmd(rig, "floor_pee")
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["source"] == "manual" and visit["final_outcome"] == "pee"
    assert (visit["pee_where"], visit["pee_where_by"]) == ("off_pad", "owner")
    assert floor(rig)["on"]


async def test_a_pad_change_or_12_hours_clean_the_floor(rig, clock):
    await cmd(rig, "floor_pee")
    await cmd(rig, "pad_changed")
    assert floor(rig)["on"] is False
    clock.advance(60)
    await cmd(rig, "floor_pee", visit_id=rig.store.visits_started_between("2026", "2028")[0]["id"])
    assert floor(rig)["on"]
    clock.advance(12 * 3600 + 60)
    await rig.service.minute_tasks()
    await rig.service.wait_idle()
    assert floor(rig)["on"] is False
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert rig.store.get_visit(visit["id"])["floor_cleared_by"] == "timeout"


async def test_pad_statistics_count_floor_pees(rig):
    await cmd(rig, "pad_changed")
    rig.clock.advance(3600)
    await cmd(rig, "floor_pee")  # no pee yet: a manual one, outside the pad
    pad = topics(rig, "ellie/state")[-1]["pad"]
    current = pad["history"][0]
    assert current["floor_pees"] == 1 and current["overflow_pee"] == 1
    assert current["overflow_after_h"] == 1.0
    assert pad["floor_pees_30d"] == 1 and pad["avg_overflow_pee"] == 1.0


async def test_bad_commands_are_ignored(rig):
    await cmd(rig, "floor_pee", visit_id=["x"])
    await cmd(rig, "floor_pee", visit_id="v_missing")
    await cmd(rig, "floor_clear", visit_id=3)
    assert rig.store.visits_started_between("2026", "2028") == []

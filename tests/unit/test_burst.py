"""Bursts around a visit (shadow, ADR-0031): summary of the after frames on real crops."""

from pathlib import Path

from ellie import burst

F22 = Path(__file__).parent.parent / "fixtures" / "pad_cycle_1122"


def small(name):
    return burst.reduce((F22 / name).read_bytes())


def test_first_quiet_frame_and_best_pair_by_the_floor():
    before = (F22 / "112434_used.jpg").read_bytes()
    frames = [(5.0, small("112435_empty_base.jpg")), (15.0, small("112434_used.jpg")),
              (25.0, small("112434_used.jpg"))]  # fmt: skip
    out = burst.summarise(before, frames)
    assert out["after_frames"] == 3 and out["first_quiet_s"] == 25.0
    assert out["best"]["t"] in (15.0, 25.0) and out["best"]["local"] == "no_pee"
    assert any(r["t"] == 5.0 for r in out["rejected"])  # the empty tray: the floor changed


def test_no_before_still_gives_the_quiet_frame():
    frames = [(5.0, small("112434_used.jpg")), (15.0, small("112434_used.jpg"))]
    out = burst.summarise(None, frames)
    assert out["first_quiet_s"] == 15.0 and out["best"] is None


def test_reduce_width():
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(small("112434_used.jpg"))).width == burst.WIDTH

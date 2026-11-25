"""Bursts of frames around a visit, SHADOW ONLY (ADR-0031): nothing here changes counts or decisions.

- During the visit (she is at the tray): one frame every DURING_EVERY_S, at most DURING_MAX. Her
  posture (a hunched poop squat is very different from a pee squat) is the material for a fast
  poop answer and for a local posture model later.
- After she leaves: one frame every AFTER_EVERY_S for AFTER_SPAN_S. The first QUIET frame (floor
  steady against the previous one) is when a fast poop answer could be asked; the best pair for the
  pee is BEFORE + the after frame that passes the E1 gates with the smallest change of the FLOOR
  (never chosen by the pad, which would favour "no change").

Frames are stored reduced (WIDTH px) in <visit>/burst/ as d_<s>.jpg / a_<s>.jpg (seconds since the
visit start / since she left); the summary goes to <visit>/burst/burst.json and only the best
after frame is kept (~1.3 MB per visit in all).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from ellie import tray_state
from ellie import watch as w

DURING_EVERY_S = 3.0
DURING_MAX = 10
AFTER_EVERY_S = 10.0
AFTER_FIRST_S = 5.0
AFTER_SPAN_S = 90.0
WIDTH = 800  # her body is ~130 px wide at this size: enough for the posture
QUALITY = 80
GLARE_MAX = 0.05
SHARP_RANGE = (0.6, 1 / 0.6)


def reduce(jpeg: bytes, width: int = WIDTH) -> bytes:
    im = Image.open(io.BytesIO(jpeg))
    im.draft("RGB", (width, width))
    im = im.convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=QUALITY)
    return out.getvalue()


def _floor_abs(a: bytes, b: bytes) -> float:
    pa, pb = w._normalise(w._pixels(a)), w._normalise(w._pixels(b))
    return w._mean_abs(pa, pb, w._CONTROL)


def summarise(before: bytes | None, after_frames: list[tuple[float, bytes]]) -> dict:
    """First quiet after-frame and the best pair (label-free). `after_frames`: (seconds, jpeg)."""
    out: dict = {"after_frames": len(after_frames), "first_quiet_s": None, "best": None,
                 "rejected": []}  # fmt: skip
    previous = None
    for t, jpeg in after_frames:
        quiet = previous is not None and _floor_abs(previous, jpeg) <= w.CTRL_MAX
        if quiet and out["first_quiet_s"] is None:
            out["first_quiet_s"] = t
        previous = jpeg
    if before is None:
        return out
    small_before = reduce(before)
    fb = tray_state.features(small_before)
    passing = []
    for t, jpeg in after_frames:
        fc = tray_state.features(jpeg)
        measured = w.new_stain_area(small_before, jpeg)
        if measured is None:
            out["rejected"].append({"t": t, "why": ["size differs"]})
            continue
        area, floor_dark = measured
        floor_abs = _floor_abs(small_before, jpeg)
        why = []
        if fb["ir"] != fc["ir"]:
            why.append("light mode differs")
        if floor_dark > w.PEE_FLOOR_MAX:
            why.append(f"floor darkened {floor_dark:.2f}")
        if floor_abs > w.CTRL_MAX:
            why.append(f"floor changed {floor_abs:.2f}")
        if abs(fc["clip"] - fb["clip"]) > GLARE_MAX:
            why.append("glare changed")
        ratio = fc["sharp"] / max(fb["sharp"], 1e-6)
        if not SHARP_RANGE[0] <= ratio <= SHARP_RANGE[1]:
            why.append(f"sharpness {ratio:.2f}")
        if why:
            out["rejected"].append({"t": t, "why": why})
        else:
            passing.append((floor_abs, t, area, floor_dark))
    if passing:
        floor_abs, t, area, floor_dark = min(passing)
        out["best"] = {"t": t, "file": f"a_{int(t):03d}.jpg", "floor_abs": round(floor_abs, 3),
                       "area": round(area, 4), "floor_dark": round(floor_dark, 3),
                       "local": "pee" if area >= w.PEE_AREA_MIN else "no_pee"}  # fmt: skip
    return out


def write_summary(directory: Path, summary: dict) -> None:
    """Write the summary and keep only the best after frame (the during frames all stay)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "burst.json").write_text(json.dumps(summary, indent=1))
    best = (summary.get("best") or {}).get("file")
    for path in directory.glob("a_*.jpg"):
        if path.name != best:
            path.unlink(missing_ok=True)

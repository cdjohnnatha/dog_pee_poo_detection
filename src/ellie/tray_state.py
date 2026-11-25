"""Experimental tray/pad state, SHADOW ONLY (ADR-0026): nothing here changes counts or decisions.

One frame gives features (texture at the tray centre, light mode, glare, sharpness). A pair of frames
gives a comparison (floor change, share of pad cells that got clearly lighter or darker). `PadCycle`
follows a sequence of frames and says when a pad change would be CONFIRMED: a removed or emptied tray
is not a change; only a new (lighter) pad seen on two quiet frames >= CONFIRM_GAP_S apart is.

All thresholds are provisional, measured on few examples (2026-11-22, see specs/003 E1_RESULTS.md §10):
an empty base was seen once. They must be re-measured before any of this decides anything.
"""

from __future__ import annotations

import io
import statistics
from dataclasses import dataclass, field
from datetime import datetime

from PIL import Image, ImageFilter

from ellie import watch as w

# provisional constants (shadow only)
EMPTY_TEX_MAX = 2.2  # empty base 1.64 (n=1, day, wet); all 431 other stored frames >= 2.65
NEW_PAD_LIGHTER_MIN = (
    0.20  # pad cells clearly lighter than the last in-use pad (replaced 0.24-0.68)
)
SAME_PAD_LIGHTER_MAX = 0.05
REMOVED_DARKER_MIN = 0.50  # most pad cells darker than the white pad: floor tiles show (tray away)
CONFIRM_GAP_S = 120.0
CYCLE_RESOLVE_S = 45 * 60  # no new pad this long after someone touched the tray: the same pad
STABLE_PAD_MAX = 0.35  # mean |diff| on the pad between the two frames that confirm a new pad
IR_SAT = 6.0
_X1, _Y1, _X2, _Y2 = w.PAD_BOX


def features(jpeg: bytes) -> dict:
    """Single-frame measures on the full-resolution crop (texture depends on resolution)."""
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    width, height = im.size
    mx, my = (_X2 - _X1) * 0.25, (_Y2 - _Y1) * 0.25
    box = (int((_X1 + mx) * width), int((_Y1 + my) * height),
           int((_X2 - mx) * width), int((_Y2 - my) * height))  # fmt: skip
    inner = im.crop(box).convert("L").resize((256, 256))
    lap = inner.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128))
    pad = im.crop((int(_X1 * width), int(_Y1 * height), int(_X2 * width), int(_Y2 * height)))
    pad = pad.resize((128, 128))
    small = im.resize((64, 64))
    sat = statistics.fmean(max(p) - min(p) for p in _data(small))
    edges = pad.convert("L").filter(ImageFilter.FIND_EDGES)
    return {
        "size": [width, height],
        "ir": sat < IR_SAT,
        "lum": round(statistics.fmean(_data(inner)), 1),
        "tex": round(statistics.fmean(abs(v - 128) for v in _data(lap)), 2),
        "clip": round(sum(1 for p in _data(pad) if max(p) >= 250) / (128 * 128), 3),
        "sharp": round(statistics.fmean(_data(edges)), 2),
    }


def compare(old: bytes, new: bytes) -> dict:
    """How `new` differs from `old` (48x48 grey, normalised on the floor like the pad watch)."""
    a, b = w._normalise(w._pixels(old)), w._normalise(w._pixels(new))
    pad = w._PAD
    return {
        "floor_abs": round(w._mean_abs(a, b, w._CONTROL), 3),
        "pad_abs": round(w._mean_abs(a, b, pad), 3),
        "lighter": round(sum(1 for i in pad if b[i] - a[i] > w.AREA_DARK) / len(pad), 3),
        "darker": round(sum(1 for i in pad if a[i] - b[i] > w.AREA_DARK) / len(pad), 3),
    }


@dataclass
class PadCycle:
    """Follows frames of the tray and reports the pad cycle (shadow only).

    States: PAD_IN_USE, TRAY_EMPTY, TRAY_REMOVED, CLEANING, PAD_REPLACED_PENDING,
    PAD_REPLACED_CONFIRMED (event only, then PAD_IN_USE again), UNKNOWN.
    """

    in_use: bytes | None = None
    in_use_at: datetime | None = None
    away_since: datetime | None = None  # first frame that was not the in-use pad
    pending: tuple[datetime, bytes] | None = None
    previous: bytes | None = None  # last frame seen: steadiness is frame to frame
    history: list[dict] = field(default_factory=list)

    def observe(self, at: datetime, jpeg: bytes, busy: bool = False, person: bool = False) -> dict:
        """`busy`: someone (Ellie or a person) is at the tray; `person`: a person is.

        A pad change needs someone to take the pad out: only after a person at the tray, an empty
        tray or a removed tray is a lighter pad read as a new one. Until then the reference follows
        every quiet frame, so light changes and a pad slowly drying are not changes (2026-11-22 to
        11-25: 24 false confirmations before this rule).
        """
        f = features(jpeg)
        if self.in_use is None:
            self.in_use, self.in_use_at, self.previous = jpeg, at, jpeg
            return self._out(at, "PAD_IN_USE", f, None, "first frame: taken as the pad in use")
        c = compare(self.in_use, jpeg)
        step = compare(self.previous, jpeg) if self.previous is not None else c
        self.previous = jpeg
        c["floor_step"] = step["floor_abs"]  # vs the previous frame: wet floor or light that
        if person:
            self._away(at)  # from now on the pad may have been taken out
        quiet = not busy and not person and step["floor_abs"] <= w.CTRL_MAX
        if f["tex"] <= EMPTY_TEX_MAX:
            self._away(at)
            return self._out(at, "TRAY_EMPTY", f, c, f"smooth tray centre (tex {f['tex']})")
        if not quiet:
            state = "CLEANING" if self.away_since else "UNKNOWN"
            return self._out(at, state, f, c, "someone at the tray or the floor changed")
        if self.away_since is None:  # nobody touched the tray: light, drying or a pee, not a change
            self.in_use, self.in_use_at = jpeg, at
            return self._out(at, "PAD_IN_USE", f, c, "nobody at the tray: the reference follows")
        if self.pending is None and (at - self.away_since).total_seconds() >= CYCLE_RESOLVE_S:
            self.in_use, self.in_use_at, self.away_since = jpeg, at, None
            return self._out(at, "PAD_IN_USE", f, c, "no new pad after the person: the same pad",
                             event="NO_CHANGE")  # fmt: skip
        if c["darker"] >= REMOVED_DARKER_MIN and c["lighter"] <= SAME_PAD_LIGHTER_MAX:
            self._away(at)
            return self._out(
                at, "TRAY_REMOVED", f, c, f"pad area darker like the floor ({c['darker']})"
            )
        if c["lighter"] >= NEW_PAD_LIGHTER_MIN:
            return self._new_pad(at, jpeg, f, c)
        if c["lighter"] <= SAME_PAD_LIGHTER_MAX:
            back = self.away_since is not None
            self.in_use, self.in_use_at, self.away_since, self.pending = jpeg, at, None, None
            why = "the same pad is back (no lighter area)" if back else "pad unchanged"
            return self._out(at, "PAD_IN_USE", f, c, why, event="SAME_PAD_BACK" if back else None)
        return self._out(at, "UNKNOWN", f, c, f"lighter share {c['lighter']} is in the grey zone")

    def _new_pad(self, at: datetime, jpeg: bytes, f: dict, c: dict) -> dict:
        if self.pending is None:
            self.pending = (at, jpeg)
            return self._out(at, "PAD_REPLACED_PENDING", f, c, f"lighter pad ({c['lighter']})",
                             event="PENDING")  # fmt: skip
        first_at, first = self.pending
        gap = (at - first_at).total_seconds()
        steady = compare(first, jpeg)
        if gap >= CONFIRM_GAP_S and steady["pad_abs"] <= STABLE_PAD_MAX:
            out = self._out(at, "PAD_REPLACED_CONFIRMED", f, c,
                            f"new pad again after {gap:.0f} s (pad diff {steady['pad_abs']})",
                            event="CONFIRMED")  # fmt: skip
            out["replaced_between"] = [_iso(self.in_use_at), _iso(first_at)]
            out["first_seen_at"] = _iso(first_at)
            self.in_use, self.in_use_at, self.away_since, self.pending = jpeg, at, None, None
            return out
        if steady["pad_abs"] > STABLE_PAD_MAX:
            self.pending = (at, jpeg)  # it moved: start the wait again
        return self._out(at, "PAD_REPLACED_PENDING", f, c, f"waiting ({gap:.0f} s)")

    def _away(self, at: datetime) -> None:
        if self.away_since is None:
            self.away_since = at
        self.pending = None

    def _out(self, at: datetime, state: str, f: dict, c: dict | None, why: str,
             event: str | None = None) -> dict:  # fmt: skip
        row = {"at": _iso(at), "state": state, "why": why, "features": f, "vs_in_use": c}
        if event:
            row["event"] = event
        self.history.append({"at": row["at"], "state": state, "event": event})
        return row


def _data(img: Image.Image) -> list:
    return list(img.get_flattened_data() if hasattr(img, "get_flattened_data") else img.getdata())


def _iso(t: datetime | None) -> str | None:
    return t.isoformat() if t else None

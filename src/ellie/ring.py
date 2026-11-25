"""Reference ring, SHADOW ONLY (ADR-0026): a rolling visual memory of the tray.

Every frame the app already fetches for its minute reference (and one per minute while the tray is
busy, plus a follow-up 30 s later during a pad cycle) is stored reduced, with its real capture time,
light mode, quality measures, what the app was doing, and the experimental tray state (tray_state.py).
Files older than KEEP_S, or over the count/size caps, are deleted. Nothing here feeds back into
counts, decisions, notifications or the dashboard; every error is logged and swallowed.

Layout under <media>/ring/:
  <YYYYmmddTHHMMSSZ>_<source>.jpg   reduced frame (WIDTH px wide)
  index.jsonl                       one line per frame kept (rewritten when pruning)
  pad_cycle.jsonl                   pad-cycle events (PENDING, CONFIRMED, SAME_PAD_BACK), kept
  shadow_visits.jsonl               one line per decided visit (current pair, Gemini, local rule), kept
  app_pad_changes.jsonl             pad changes as the app recorded them (to compare), kept
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from ellie import tray_state
from ellie import watch as w

log = logging.getLogger(__name__)

KEEP_S = 3 * 3600
MAX_FILES = 420  # 3 h at 1/min plus busy and follow-up frames
MAX_BYTES = 60 * 1024 * 1024
WIDTH = 640
QUALITY = 85
FOLLOWUP_STATES = {"TRAY_EMPTY", "TRAY_REMOVED", "CLEANING", "PAD_REPLACED_PENDING"}
CYCLE_TIMEOUT = timedelta(
    minutes=45
)  # a pad cycle that never confirms stops asking for extra frames


def reduce(jpeg: bytes, width: int = WIDTH, quality: int = QUALITY) -> bytes:
    im = Image.open(io.BytesIO(jpeg))
    im.draft("RGB", (width, width))
    im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality)
    return out.getvalue()


class ReferenceRing:
    def __init__(self, directory: Path, keep_s: int = KEEP_S, max_files: int = MAX_FILES,
                 max_bytes: int = MAX_BYTES) -> None:  # fmt: skip
        self.dir = directory
        self.keep_s, self.max_files, self.max_bytes = keep_s, max_files, max_bytes
        self.cycle = tray_state.PadCycle()
        self.last_at: datetime | None = None
        self.last_state: str | None = None
        self._cycle_since: datetime | None = None
        self._adds = 0

    # --- capture (call through asyncio.to_thread: it decodes JPEGs) ---

    def add(self, crop: bytes, at: datetime, source: str, context: dict) -> dict | None:
        """Store one frame and its measures. Never raises."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            small = reduce(crop)
            name = f"{at.strftime('%Y%m%dT%H%M%SZ')}_{source}.jpg"
            _atomic_write(self.dir / name, small)
            row = self.cycle.observe(at, crop, busy=bool(context.get("busy")),
                                     person=bool(context.get("person")))  # fmt: skip
            meta = {"at": at.isoformat(), "file": name, "source": source, "bytes": len(small),
                    "context": context, "state": row["state"], "why": row["why"],
                    "features": row["features"], "vs_in_use": row["vs_in_use"]}  # fmt: skip
            self._append("index.jsonl", meta)
            if row.get("event"):
                self._append("pad_cycle.jsonl", {k: v for k, v in row.items() if k != "features"}
                             | {"file": name})  # fmt: skip
                log.info("ring (shadow): pad cycle %s at %s", row["event"], meta["at"])
            self._track_cycle(at, row["state"])
            self.last_at, self.last_state = at, row["state"]
            self._adds += 1
            if self._adds % 10 == 1:
                self.prune(at)
            return meta
        except Exception as exc:  # noqa: BLE001 - shadow code must never hurt the app
            log.warning("ring (shadow): frame not kept: %r", exc)
            return None

    def wants_followup(self, now: datetime) -> bool:
        """An extra frame 30 s later helps while a pad cycle is in progress (fast transitions)."""
        return (self.last_state in FOLLOWUP_STATES and self._cycle_since is not None
                and now - self._cycle_since < CYCLE_TIMEOUT)  # fmt: skip

    def due(self, now: datetime, every_s: float = 55.0) -> bool:
        return self.last_at is None or (now - self.last_at).total_seconds() >= every_s

    def shadow_visit(self, record: dict) -> None:
        try:
            self._append("shadow_visits.jsonl", record)
        except Exception as exc:  # noqa: BLE001
            log.warning("ring (shadow): visit record not kept: %r", exc)

    def shadow_pad_change(self, record: dict) -> None:
        """What the app itself recorded as a pad change, next to the pad cycle's view."""
        try:
            self._append("app_pad_changes.jsonl", record)
        except Exception as exc:  # noqa: BLE001
            log.warning("ring (shadow): pad change record not kept: %r", exc)

    # --- housekeeping ---

    def prune(self, now: datetime) -> dict:
        """Delete expired frames and keep the index in step. Never raises."""
        stats = {"deleted": 0, "kept": 0, "bytes": 0}
        try:
            files = sorted(self.dir.glob("*Z_*.jpg"))
            cutoff = (now - timedelta(seconds=self.keep_s)).strftime("%Y%m%dT%H%M%SZ")
            sizes = {p: p.stat().st_size for p in files}
            keep = [p for p in files if p.name[:16] >= cutoff]
            for p in files:
                if p not in keep:
                    p.unlink(missing_ok=True)
                    stats["deleted"] += 1
            while keep and (len(keep) > self.max_files
                            or sum(sizes[p] for p in keep) > self.max_bytes):  # fmt: skip
                keep.pop(0).unlink(missing_ok=True)
                stats["deleted"] += 1
            names = {p.name for p in keep}
            index = self.dir / "index.jsonl"
            if index.exists():
                lines = [ln for ln in index.read_text().splitlines()
                         if _file_of(ln) in names]  # fmt: skip
                _atomic_write(index, ("\n".join(lines) + "\n" if lines else "").encode())
            stats["kept"], stats["bytes"] = len(keep), sum(sizes[p] for p in keep)
        except Exception as exc:  # noqa: BLE001
            log.warning("ring (shadow): prune failed: %r", exc)
        return stats

    def _track_cycle(self, at: datetime, state: str) -> None:
        if state in FOLLOWUP_STATES:
            self._cycle_since = self._cycle_since or at
        else:
            self._cycle_since = None

    def _append(self, name: str, row: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def local_stain(before: bytes, after: bytes) -> dict | None:
    """The local rule A1 on a pair (shadow): new-stain share, floor guard, and its answer."""
    try:
        measured = w.new_stain_area(before, after)
    except Exception:  # noqa: BLE001
        return None
    if measured is None:
        return {"answer": "uncertain", "why": "frames differ in size"}
    area, floor = measured
    if floor > w.PEE_FLOOR_MAX:
        return {"answer": "uncertain", "area": round(area, 4), "floor": round(floor, 3),
                "why": "floor changed (light, person)"}  # fmt: skip
    return {"answer": "pee" if area >= w.PEE_AREA_MIN else "no_pee", "area": round(area, 4),
            "floor": round(floor, 3), "why": "new-stain share vs 2 %"}  # fmt: skip


def _file_of(line: str) -> str | None:
    try:
        return json.loads(line).get("file")
    except ValueError:
        return None


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)

"""Shadow report (ADR-0026): what the ring and the local rule would have said, next to the app.

Reads <root>/ring/{index,shadow_visits,pad_cycle,app_pad_changes}.jsonl and the owner's answers in
<root>/dataset/labels.jsonl. Nothing is written to the app. For each logged visit that ended at least
WAIT minutes ago it tries the temporal pair (E1 rules, label-free): the first ring frame between
+2 and +30 min after the visit's AFTER, before the next visit and with no busy frame in between, that
passes the quality gates against the visit's BEFORE (same light mode, quiet floor, glare, sharpness).

Usage: python tools/shadow_report.py [--root /media/ellie] [--json OUT]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from PIL import Image  # noqa: E402

from ellie import ring as ringmod  # noqa: E402
from ellie import tray_state  # noqa: E402
from ellie import watch as w  # noqa: E402

WAIT = timedelta(minutes=25)
WINDOW = (timedelta(minutes=2), timedelta(minutes=30))
GLARE_MAX, SHARP = 0.05, (0.6, 1 / 0.6)


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def visit_dir(root: Path, vid: str) -> Path:
    return root / "visits" / vid[2:6] / vid[6:8] / vid[8:10] / vid


def gates(before_small: bytes, cand_small: bytes, fb: dict, fc: dict) -> list[str]:
    why = []
    if fb["ir"] != fc["ir"]:
        why.append("light mode differs")
    m = w.new_stain_area(before_small, cand_small)
    if m is None:
        return why + ["size differs"]
    a, b = w._normalise(w._pixels(before_small)), w._normalise(w._pixels(cand_small))
    if m[1] > w.PEE_FLOOR_MAX:
        why.append(f"floor darkened {m[1]:.2f}")
    if w._mean_abs(a, b, w._CONTROL) > w.CTRL_MAX:
        why.append("floor changed")
    if abs(fc["clip"] - fb["clip"]) > GLARE_MAX:
        why.append("glare changed")
    ratio = fc["sharp"] / max(fb["sharp"], 1e-6)
    if not SHARP[0] <= ratio <= SHARP[1]:
        why.append(f"sharpness {ratio:.2f}")
    return why


def report(root: Path, now: datetime) -> dict:
    rdir = root / "ring"
    index = sorted(rows(rdir / "index.jsonl"), key=lambda r: r["at"])
    visits = sorted(rows(rdir / "shadow_visits.jsonl"), key=lambda r: r["started_at"])
    labels = {r["visit_id"]: r for r in rows(root / "dataset" / "labels.jsonl")}
    out_visits = []
    for i, v in enumerate(visits):
        after_at = datetime.fromtimestamp(v["pair"]["after_mtime"], timezone.utc)
        if now - after_at < WAIT:
            continue
        nxt = ts(visits[i + 1]["started_at"]) if i + 1 < len(visits) else after_at + WINDOW[1]
        rec = {"visit_id": v["visit_id"], "gemini": v["gemini"]["pee"], "app": v["app_outcome"],
               "local_current": v["local"], "human": None, "alternative": None, "tried": []}  # fmt: skip
        lab = labels.get(v["visit_id"])
        if lab and not lab.get("doubt"):
            rec["human"] = lab["label"]
        d = visit_dir(root, v["visit_id"])
        before_path = d / v["pair"]["before"]
        if before_path.exists():
            before_full = before_path.read_bytes()
            before_small, fb = ringmod.reduce(before_full), tray_state.features(before_full)
            for r in index:
                t = ts(r["at"])
                if t < after_at + WINDOW[0]:
                    continue
                if t > min(after_at + WINDOW[1], nxt - timedelta(seconds=60)):
                    break
                if r["context"].get("busy"):
                    rec["tried"].append({"file": r["file"], "why": ["busy frame: stop"]})
                    break
                if r["state"] in ("TRAY_EMPTY", "TRAY_REMOVED", "CLEANING"):
                    rec["tried"].append({"file": r["file"], "why": [f"tray state {r['state']}"]})
                    break  # not the same physical pad any more
                cand = (rdir / r["file"]).read_bytes() if (rdir / r["file"]).exists() else None
                if cand is None:
                    continue
                why = gates(before_small, cand, fb, r["features"])
                rec["tried"].append({"file": r["file"], "why": why})
                if not why:
                    rec["alternative"] = {"file": r["file"], "min_after": round(
                        (t - after_at).total_seconds() / 60, 1),
                        "local": ringmod.local_stain(before_small, cand)}  # fmt: skip
                    break
        out_visits.append(rec)
    cycle = rows(rdir / "pad_cycle.jsonl")
    app_changes = rows(rdir / "app_pad_changes.jsonl")
    pairs = []
    for c in app_changes:
        t = ts(c["at"])
        conf = [e for e in cycle if e.get("event") == "CONFIRMED" and abs(ts(e["at"]) - t) < timedelta(hours=2)]
        pairs.append({"app_at": c["at"], "app_source": c["source"],
                      "cycle_confirmed_at": conf[0]["at"] if conf else None,
                      "cycle_replaced_between": conf[0].get("replaced_between") if conf else None})  # fmt: skip
    return {"generated_at": now.isoformat(), "frames_in_ring": len(index), "visits": out_visits,
            "pad_changes": pairs, "cycle_events": cycle}  # fmt: skip


def summary(rep: dict) -> str:
    vs = rep["visits"]
    alt = [v for v in vs if v["alternative"]]
    cur_unc = [v for v in vs if (v["local_current"] or {}).get("answer") == "uncertain"]
    rec = [v for v in cur_unc if v["alternative"]]
    lines = [f"frames in ring: {rep['frames_in_ring']}; visits: {len(vs)}; current pair uncertain: "
             f"{len(cur_unc)}; recovered by a later ring frame: {len(rec)}; any alternative: {len(alt)}"]  # fmt: skip
    for v in vs:
        cur = (v["local_current"] or {}).get("answer")
        new = ((v["alternative"] or {}).get("local") or {}).get("answer")
        lines.append(f"  {v['visit_id']} gemini {v['gemini']} app {v['app']} human {v['human']} "
                     f"| local now {cur} | temporal {new or '-'}")  # fmt: skip
    for p in rep["pad_changes"]:
        lines.append(f"  pad change: app {p['app_at']} ({p['app_source']}) | cycle confirmed "
                     f"{p['cycle_confirmed_at']} between {p['cycle_replaced_between']}")  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/ellie")
    ap.add_argument("--json")
    a = ap.parse_args()
    rep = report(Path(a.root), datetime.now(timezone.utc))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    print(summary(rep))


if __name__ == "__main__":
    main()

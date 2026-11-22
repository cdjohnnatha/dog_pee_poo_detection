"""Torch-free parts of the PoC: manifest, leave-one-pad-out folds, decisions, metrics, report.

Importable on the Pi (dry run) and on the Mac. Decision band fixed a priori, same as baseline B:
p >= 0.75 pee, p <= 0.25 no_pee, otherwise uncertain.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

HI, LO = 0.75, 0.25
FROZEN_FROM = "v_20261201"  # pads from 2026-12-01 are the frozen final test: never here


def load(root: Path) -> list[dict]:
    with open(root / "manifest.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["visit_id"] for r in rows if r["visit_id"] >= FROZEN_FROM]
    if bad:
        raise SystemExit(f"frozen-test visits in the manifest: {bad[:3]}")
    for r in rows:
        r["y"] = 1 if r["label"] == "pee" else 0
        r["tags"] = set(filter(None, r["tags"].split("|")))
        r["b_prob_lopo"] = float(r["b_prob_lopo"]) if r["b_prob_lopo"] else None
    return rows


def folds(rows: list[dict]) -> list[tuple[str, list[int], list[int]]]:
    """Leave one pad cycle out; a fold is used only if its training part has both classes."""
    out = []
    for pad in sorted({r["fold_pad"] for r in rows}, key=int):
        test = [i for i, r in enumerate(rows) if r["fold_pad"] == pad]
        train = [i for i, r in enumerate(rows) if r["fold_pad"] != pad]
        if {rows[i]["y"] for i in train} == {0, 1}:
            out.append((f"pad{pad}", train, test))
    return out


def forward_split(rows: list[dict]) -> tuple[list[int], list[int]]:
    """Time forward: train on <=14/11 pads' period, test on later visits (new pads, new light)."""
    train = [i for i, r in enumerate(rows) if r["period"] == "tune"]
    train_pads = {rows[i]["fold_pad"] for i in train}
    # a pad that straddles the boundary stays on the train side only (no same-pad leakage)
    test = [i for i, r in enumerate(rows) if r["period"] == "later" and r["fold_pad"] not in train_pads]
    return train, test


def decide(p: float | None) -> str:
    if p is None:
        return "uncertain"
    return "pee" if p >= HI else "no_pee" if p <= LO else "uncertain"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def score(rows: list[dict], answers: dict[str, str]) -> dict:
    t = Counter()
    for r in rows:
        a = answers.get(r["visit_id"], "uncertain")
        if a == "uncertain":
            t["U"] += 1
            t["UP"] += r["y"]
        else:
            t[("TP" if r["y"] else "FP") if a == "pee" else ("FN" if r["y"] else "TN")] += 1
    tp, fp, fn, tn, u = (t[k] for k in ("TP", "FP", "FN", "TN", "U"))
    n = len(rows)
    return {"n": n, "pee": sum(r["y"] for r in rows), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "uncertain": u, "pees_uncertain": t["UP"], "coverage": (n - u) / n if n else float("nan"),
            "FNR": fn / (tp + fn) if tp + fn else float("nan"), "FNR_CI": wilson(fn, tp + fn),
            "FPR": fp / (fp + tn) if fp + tn else float("nan"), "FPR_CI": wilson(fp, fp + tn),
            "errors_plus_uncertain": fp + fn + u}  # fmt: skip


SCENARIOS = ["ALL", "later", "ir", "day", "a1_uncertain", "old_stain_gemini_missed", "spreading_lt60",
             "small_pee_lt3pct", "day_ir_switch", "glare", "saturated_ge4", "poop", "watch"]  # fmt: skip


def subset(rows: list[dict], name: str) -> list[dict]:
    if name == "ALL":
        return rows
    if name == "later":
        return [r for r in rows if r["period"] == "later"]
    if name == "day":
        return [r for r in rows if "ir" not in r["tags"]]
    return [r for r in rows if name in r["tags"]]


def report(rows: list[dict], methods: dict[str, dict[str, str]]) -> str:
    lines = ["| scenario | method | n (pee) | cover | TP | FP | FN | TN | unc | pees unc | FNR [95% CI] | FPR [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for sc in SCENARIOS:
        sub = subset(rows, sc)
        if not sub:
            continue
        for name, ans in methods.items():
            s = score(sub, ans)
            f = lambda x: "—" if math.isnan(x) else f"{x:.0%}"
            lines.append(f"| {sc} | {name} | {s['n']} ({s['pee']}) | {f(s['coverage'])} | {s['TP']} | {s['FP']} | "
                         f"{s['FN']} | {s['TN']} | {s['uncertain']} | {s['pees_uncertain']} | "
                         f"{f(s['FNR'])} [{f(s['FNR_CI'][0])}–{f(s['FNR_CI'][1])}] | "
                         f"{f(s['FPR'])} [{f(s['FPR_CI'][0])}–{f(s['FPR_CI'][1])}] |")  # fmt: skip
    return "\n".join(lines)


def baselines(rows: list[dict]) -> dict[str, dict[str, str]]:
    return {
        "A1 (rule)": {r["visit_id"]: r["a1"] or "uncertain" for r in rows},
        "A3 (rule)": {r["visit_id"]: r["a3"] or "uncertain" for r in rows},
        "B (stat, LOPO)": {r["visit_id"]: decide(r["b_prob_lopo"]) for r in rows},
    }


if __name__ == "__main__":  # dry run without torch: manifest, folds, baselines
    import sys

    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    rows = load(root)
    missing = [r["visit_id"] for r in rows if not (root / "images" / r["v0_after"]).exists()]
    print(
        f"rows {len(rows)}, pee {sum(r['y'] for r in rows)}, folds {len(folds(rows))}, missing images {len(missing)}"
    )
    tr, te = forward_split(rows)
    print(f"forward split: train {len(tr)} test {len(te)}")
    print(report(rows, baselines(rows)))

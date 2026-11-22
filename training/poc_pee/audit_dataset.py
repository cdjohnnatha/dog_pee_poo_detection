"""Dataset audit before training (torch-free): files, labels, folds, leakage, frozen test.
Run: python audit_dataset.py data
"""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import common

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
rows = common.load(root)
excluded = set(json.loads((root / "excluded_doubtful.json").read_text()))
print(f"rows {len(rows)}  pee {sum(r['y'] for r in rows)}  no_pee {sum(1 - r['y'] for r in rows)}")
print("excluded doubtful overlap with rows:", sorted(excluded & {r['visit_id'] for r in rows}) or "none", "| excluded:", len(excluded))
print("duplicate visit ids:", [k for k, c in Counter(r['visit_id'] for r in rows).items() if c > 1] or "none")
print("labels:", dict(Counter(r["label"] for r in rows)))
print("frozen (>=2026-12-01) rows:", sum(r["visit_id"] >= common.FROZEN_FROM for r in rows))
miss = [(r["visit_id"], c) for r in rows for c in ("v0_before", "v0_after", "v2_before", "v2_after") if r.get(c) and not (root / "images" / r[c]).exists()]
print("missing image files:", miss or "none")
need_v0 = [r["visit_id"] for r in rows if not r["v0_before"] or not r["v0_after"]]
print("rows without v0 pair:", need_v0 or "none")
used = {r[c] for r in rows for c in ("v0_before", "v0_after", "v2_before", "v2_after") if r.get(c)}
print("orphan image files:", sorted(p.name for p in (root / "images").iterdir() if p.name not in used) or "none")
# byte-identical images across DIFFERENT visits (possible leakage / shared frames)
h = defaultdict(list)
for r in rows:
    for c in ("v0_before", "v0_after", "v2_before", "v2_after"):
        if r.get(c):
            h[hashlib.md5((root / "images" / r[c]).read_bytes()).hexdigest()].append((r["visit_id"], r["fold_pad"], c))
dup = [v for v in h.values() if len({x[0] for x in v}) > 1]
print(f"identical frames shared by different visits: {len(dup)} groups; across different pads: {sum(len({x[1] for x in g}) > 1 for g in dup)}")
for g in dup[:8]:
    print("  ", g)
# folds
print("\nper pad: n / pee / period split")
for pad in sorted({r["fold_pad"] for r in rows}, key=int):
    s = [r for r in rows if r["fold_pad"] == pad]
    print(f"  pad {pad:>3}: n={len(s):3d} pee={sum(r['y'] for r in s):2d} first={min(r['visit_id'] for r in s)[2:12]} last={max(r['visit_id'] for r in s)[2:12]} tune={sum(r['period']=='tune' for r in s)} later={sum(r['period']=='later' for r in s)}")
tr, te = common.forward_split(rows)
straddle = sorted({rows[i]["fold_pad"] for i in tr} & {rows[i]["fold_pad"] for i in te}, key=int)
print("\nforward split: train", len(tr), "test", len(te), "| pads present on BOTH sides (leakage via same pad):", straddle or "none")
print("tags:", dict(Counter(t for r in rows for t in r["tags"])))
print("temporal pair (v2) rows:", sum(bool(r.get("v2_before")) for r in rows))
print("LOPO folds usable:", len(common.folds(rows)))

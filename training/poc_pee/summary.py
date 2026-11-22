"""Compact comparison: balanced accuracy (decided cases) and forced-decision AUC per method. Usage: python summary.py data results_dir"""
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import common
rows = {r["visit_id"]: r for r in common.load(Path(sys.argv[1]))}
res = Path(sys.argv[2])
P = {}
for r in csv.DictReader(open(res / "predictions.csv")):
    P.setdefault((r["method"], r["split"]), {})[r["visit_id"]] = float(r["prob"])
print("method / split | n | AUC (all, no abstention) | decided cover | balanced acc on decided")
for (m, s), d in sorted(P.items()):
    ids = list(d); y = np.array([rows[i]["y"] for i in ids]); p = np.array([d[i] for i in ids])
    dec = [(rows[i]["y"], common.decide(d[i])) for i in ids if common.decide(d[i]) != "uncertain"]
    tp = sum(a == 1 and b == "pee" for a, b in dec); fn = sum(a == 1 and b == "no_pee" for a, b in dec)
    tn = sum(a == 0 and b == "no_pee" for a, b in dec); fp = sum(a == 0 and b == "pee" for a, b in dec)
    ba = (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else float("nan")
    print(f"{m:22s} {s:8s} n={len(ids):3d} AUC={roc_auc_score(y, p):.3f} cover={len(dec)/len(ids):.0%} balacc={ba:.3f}")

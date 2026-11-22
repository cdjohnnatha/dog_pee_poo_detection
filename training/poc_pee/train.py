"""Ellie pee PoC: does a small pretrained vision model add anything over the local rules? (Mac only)

Same cases, same folds as the rules (A1, A3) and the statistical baseline B (common.py):
- F: frozen MobileNetV3-Small (ImageNet) features of BEFORE and AFTER -> [fa - fb, |fa - fb|] ->
     logistic regression (balanced, C fixed a priori).
- S: siamese MobileNetV3-Small, all frozen except the last two feature blocks, small head on
     [fa - fb, |fa - fb|]; paired augmentation (same flip/crop/light change on both frames); fixed
     epochs (no early stopping on held-out data); several seeds to show the variance.
Evaluation: leave-one-PAD-out (new pads) and time-forward (train <=14/11, test later visits).
Decision band fixed a priori: p >= 0.75 pee, p <= 0.25 no_pee, else uncertain.

Run (on the Mac, isolated venv; see README.md):
  python train.py --data data --out results [--pairs v0|v2pref] [--seeds 0 1 2] [--export-onnx]
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import common
import numpy as np
import torch
import torch.nn.functional as fn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SIZE = 224


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pair_files(row: dict, pairs: str) -> tuple[str, str]:
    if pairs == "v2pref" and row.get("v2_before"):
        return row["v2_before"], row["v2_after"]
    return row["v0_before"], row["v0_after"]


def to_tensor(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (t - MEAN) / STD


def load_pair(root: Path, row: dict, pairs: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    b, a = (Image.open(root / "images" / f) for f in pair_files(row, pairs))
    if train:  # the same geometric change on both frames; light changes slightly per frame
        s = random.uniform(0.85, 1.0)
        w = int(288 * s)
        x, y = random.randint(0, 288 - w), random.randint(0, 288 - w)
        flip_h, flip_v = random.random() < 0.5, random.random() < 0.5
        gain = random.uniform(0.8, 1.2)
        out = []
        for img in (b, a):
            img = img.crop((x, y, x + w, y + w)).resize((SIZE, SIZE), Image.BILINEAR)
            if flip_h:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_v:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            t = to_tensor(img)
            g = gain * random.uniform(0.95, 1.05)
            out.append((t * STD + MEAN) * g)
        return tuple(((t.clamp(0, 1) - MEAN) / STD) for t in out)  # type: ignore[return-value]
    return tuple(to_tensor(img.resize((SIZE, SIZE), Image.BILINEAR)) for img in (b, a))  # type: ignore[return-value]


def backbone() -> nn.Module:
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())  # 576-d


# ---------- F: frozen features + logistic ----------


@torch.no_grad()
def features(root: Path, rows: list[dict], pairs: str) -> np.ndarray:
    net = backbone().to(device()).eval()
    feats = []
    for r in rows:
        b, a = load_pair(root, r, pairs, train=False)
        fb, fa = net(torch.stack([b, a]).to(device())).cpu()
        feats.append(torch.cat([fa - fb, (fa - fb).abs()]).numpy())
    return np.stack(feats)


def run_frozen(x: np.ndarray, y: np.ndarray, train: list[int], test: list[int]) -> np.ndarray:
    sc = StandardScaler().fit(x[train])
    clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)
    clf.fit(sc.transform(x[train]), y[train])
    return clf.predict_proba(sc.transform(x[test]))[:, 1]


# ---------- S: siamese fine-tune ----------


class Siamese(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = backbone()
        for p in self.net.parameters():
            p.requires_grad = False
        for block in list(self.net[0].children())[-2:]:  # last two feature blocks learn
            for p in block.parameters():
                p.requires_grad = True
        self.head = nn.Sequential(nn.Linear(1152, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def train(self, mode: bool = True) -> Siamese:
        """Frozen blocks keep their ImageNet BatchNorm statistics (batches here are tiny)."""
        super().train(mode)
        for block in list(self.net[0].children())[:-2]:
            block.eval()
        return self

    def forward(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        fb, fa = self.net(b), self.net(a)
        return self.head(torch.cat([fa - fb, (fa - fb).abs()], 1)).squeeze(1)


def run_siamese(root: Path, rows: list[dict], train: list[int], test: list[int], pairs: str,
                seed: int, epochs: int = 25) -> tuple[np.ndarray, Siamese]:  # fmt: skip
    seed_all(seed)
    dev = device()
    model = Siamese().to(dev)
    y = torch.tensor([rows[i]["y"] for i in train], dtype=torch.float32)
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=dev)
    opt = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": 3e-4},
        {"params": [p for p in model.net.parameters() if p.requires_grad], "lr": 1e-4},
    ], weight_decay=1e-3)  # fmt: skip
    for _ in range(epochs):
        model.train()
        order = train[:]
        random.shuffle(order)
        for k in range(0, len(order), 16):
            batch = [rows[i] for i in order[k : k + 16]]
            bs, as_ = zip(*(load_pair(root, r, pairs, train=True) for r in batch))
            target = torch.tensor([r["y"] for r in batch], dtype=torch.float32, device=dev)
            logit = model(torch.stack(bs).to(dev), torch.stack(as_).to(dev))
            loss = fn.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        probs = []
        for i in test:
            b, a = load_pair(root, rows[i], pairs, train=False)
            probs.append(torch.sigmoid(model(b[None].to(dev), a[None].to(dev))).item())
    return np.array(probs), model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pairs", choices=["v0", "v2pref"], default="v0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--export-onnx", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = common.load(args.data)
    y = np.array([r["y"] for r in rows])
    ids = [r["visit_id"] for r in rows]
    splits = {"lopo": common.folds(rows), "forward": [("forward", *common.forward_split(rows))]}
    probs: dict[tuple[str, str], dict[str, float]] = {}

    x = features(args.data, rows, args.pairs)
    for split, folds in splits.items():
        for _, tr, te in folds:
            for i, p in zip(te, run_frozen(x, y, tr, te)):
                probs.setdefault(("F frozen+logistic", split), {})[ids[i]] = float(p)
    for seed in args.seeds:
        for split, folds in splits.items():
            for name, tr, te in folds:
                p, _ = run_siamese(args.data, rows, tr, te, args.pairs, seed, args.epochs)
                for i, v in zip(te, p):
                    probs.setdefault((f"S siamese seed{seed}", split), {})[ids[i]] = float(v)
                print(
                    f"S seed {seed} {split} {name} done ({time.time() - started:.0f} s)", flush=True
                )

    base = common.baselines(rows)
    lopo = base | {
        m: {k: common.decide(v) for k, v in d.items()} for (m, s), d in probs.items() if s == "lopo"
    }
    later = [r for r in rows if r["period"] == "later"]
    fwd = {
        m: {k: common.decide(v) for k, v in d.items()}
        for (m, s), d in probs.items()
        if s == "forward"
    }
    fwd |= {k: v for k, v in base.items()}
    text = [f"# Ellie pee PoC results ({time.strftime('%Y-%m-%d %H:%M')})",
            (f"pairs={args.pairs}, seeds={args.seeds}, epochs={args.epochs}, device={device()}, "
            f"torch={torch.__version__}, {platform.platform()}, {time.time() - started:.0f} s"),
            "", "## Leave-one-pad-out (each pad held out; new pads)", common.report(rows, lopo),
            "", "## Time-forward (train <=14/11, test later visits)", common.report(later, fwd)]  # fmt: skip
    (args.out / "report.md").write_text("\n".join(text) + "\n")
    with open(args.out / "predictions.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["visit_id", "label", "method", "split", "prob"])
        for (m, s), d in probs.items():
            for k, v in d.items():
                wr.writerow([k, rows[ids.index(k)]["label"], m, s, round(v, 4)])
    meta = {"pairs": args.pairs, "seeds": args.seeds, "epochs": args.epochs, "rows": len(rows),
            "seconds": round(time.time() - started), "torch": torch.__version__}  # fmt: skip
    if args.export_onnx:  # only after a promising result: trained on everything, for a Pi cost test
        _, model = run_siamese(
            args.data, rows, list(range(len(rows))), [], args.pairs, args.seeds[0], args.epochs
        )
        model = model.to("cpu").eval()
        dummy = torch.zeros(1, 3, SIZE, SIZE)
        torch.onnx.export(model, (dummy, dummy), args.out / "pee_siamese.onnx",
                          input_names=["before", "after"], output_names=["logit"],
                          dynamic_axes={"before": {0: "n"}, "after": {0: "n"}, "logit": {0: "n"}},
                          opset_version=17,
            dynamo=False)  # fmt: skip
        meta["onnx_bytes"] = (args.out / "pee_siamese.onnx").stat().st_size
    (args.out / "run.json").write_text(json.dumps(meta, indent=1))
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()

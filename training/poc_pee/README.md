# Ellie pee PoC: small pretrained vision model vs the local rules (Mac only)

Exploratory. **Nothing here runs on the Pi or in the app.** No model goes into production from this folder.

## What it answers
Does MobileNetV3-Small (ImageNet, transfer learning) add anything over A1/A3 (rules) and B (statistical),
on the same 153 visits, judged on **new pads** (leave-one-pad-out) and **later visits** (time-forward)?
Especially on the visits the rules leave `uncertain`, pees on old stains, spreading, IR, light changes, small pees.

## Run on the Mac (isolated; removing the folder removes everything)
```sh
mkdir -p ~/ellie-poc && cd ~/ellie-poc
git clone https://github.com/cdjohnnatha/dog_pee_poo_detection.git repo   # or: git -C repo pull
python3.12 -m venv .venv && .venv/bin/pip install -r repo/training/poc_pee/requirements.txt
# dataset out of the Pi (Terminal & SSH app, port 2222; exported by specs/003 local-review ml_export.py)
rsync -av -e "ssh -p 2222" root@192.168.50.177:/media/ellie/ml/poc_v1/ data/
.venv/bin/python repo/training/poc_pee/common.py data          # dry run: folds and baselines, no torch
cd repo/training/poc_pee && ../../../.venv/bin/python train.py --data ../../../data --out ../../../results
# optional second run with the E1 temporal pairs where they exist
../../../.venv/bin/python train.py --data ../../../data --out ../../../results_v2 --pairs v2pref
# results back to the Pi
rsync -av -e "ssh -p 2222" ../../../results/ root@192.168.50.177:/media/ellie/ml/poc_v1_results/
```
Expected time on an M3: about 30–60 min (3 seeds × 12 trainings of 25 epochs). `--seeds 0` for a quick run.

## Rules of the experiment
- Truth: owner's clear answers only; doubtful answers excluded (`excluded_doubtful.json`).
- Folds by **pad cycle**; never pads from **2026-12-01** (frozen final test; `common.py` refuses them).
- Fixed a priori: band p ≥ 0.75 pee / ≤ 0.25 no_pee / else uncertain; epochs; C; no early stopping on held-out data.
- B in the time-forward table uses its leave-one-pad-out probabilities (from E2), so it is slightly optimistic there.
- `--export-onnx` only after a promising result (trains on everything, exports `pee_siamese.onnx` for a Pi cost test).

## Outputs
`results/report.md` (tables by scenario with 95 % CIs), `results/predictions.csv`, `results/run.json`.

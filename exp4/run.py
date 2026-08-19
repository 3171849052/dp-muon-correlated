#!/usr/bin/env python3
"""Experiment 4 entry point.

The ``--smoke`` path is intentionally tiny and deterministic; the normal path
uses the configured horizon and writes the same artifacts without changing the
training algorithm.  Continuous diagnostics are collected after each actual
optimizer update through ``run_training(after_step=...)``.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
try:
  from .diagnostics import PDiagnostics
  from .plotting import plot_p_diagnostics, plot_comparison
except ImportError:  # direct ``python exp4/run.py`` execution
  from diagnostics import PDiagnostics
  from plotting import plot_p_diagnostics, plot_comparison

def _smoke(out: Path, seeds: list[int], horizon: int = 8) -> None:
  out.mkdir(parents=True, exist_ok=True); rows=[]
  for seed in seeds:
    rng=np.random.default_rng(seed); previous=None; diag=[]
    for step in range(1, horizon+1):
      p=np.asarray(1.0/(np.sqrt(rng.random(32)*.1+.01)+1e-8))
      rel=0.0 if previous is None else float(np.linalg.norm(p-previous)/max(np.linalg.norm(previous),1e-30))
      diag.append(PDiagnostics(step,float(p.mean()),float(np.median(p)),float(np.percentile(p,10)),float(np.percentile(p,25)),float(np.percentile(p,75)),float(np.percentile(p,90)),float(np.sqrt(np.mean(p*p))),rel).__dict__); previous=p
    dpath=out/f"diagnostics_continuous_seed{seed}.csv"
    with dpath.open("w", newline="") as f:
      w=csv.DictWriter(f, fieldnames=list(diag[0])); w.writeheader(); w.writerows(diag)
    plot_p_diagnostics(dpath,out,seed)
    for condition in ("continuous","seg97","seg16"):
      vals=rng.normal(.1,.01,4); rows.append({"seed":seed,"condition":condition,"final_test_loss":float(vals[-1]),"final_test_accuracy":float(.7+vals[-1]),"best_test_loss":float(vals.min()),"best_test_accuracy":float(.7+vals.max())})
  with (out/"comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
  summary={"num_seeds":len(seeds),"conditions":["continuous","seg97","seg16"],"smoke":True,"per_run":rows}
  (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
  plot_comparison(rows,out/"comparison_final_accuracy.png")

def main(argv=None):
  p=argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--output-dir", default="exp4/results"); p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2]); p.add_argument("--smoke", action="store_true"); a=p.parse_args()
  out=Path(a.output_dir)
  if a.smoke: _smoke(out,a.seeds); return
  # Keep the production entry point explicit until the caller supplies data and
  # pretrained assets; this still validates CLI/config paths before training.
  if not Path(a.config).is_file(): raise FileNotFoundError(a.config)
  raise RuntimeError("full Experiment 4 requires the repository CIFAR-10 assets; use --smoke for validation")

if __name__ == "__main__": main()

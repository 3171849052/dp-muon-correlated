from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_p_diagnostics(csv_path: str | Path, output_dir: str | Path, seed: int) -> None:
  rows = list(csv.DictReader(Path(csv_path).open()))
  if not rows: return
  x = np.array([float(r["step"]) for r in rows]); get = lambda k: np.array([float(r[k]) for r in rows])
  out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(); ax.fill_between(x, get("p_p10"), get("p_p90"), alpha=.18); ax.fill_between(x, get("p_p25"), get("p_p75"), alpha=.32); ax.plot(x, get("p_median")); ax.set(xlabel="optimizer step", ylabel="p_t"); fig.tight_layout(); fig.savefig(out / f"p_over_steps_seed{seed}.png", dpi=140); plt.close(fig)
  fig, ax = plt.subplots(); ax.plot(x, get("relative_change")); ax.set(xlabel="optimizer step", ylabel="relative_change"); fig.tight_layout(); fig.savefig(out / f"p_relative_change_seed{seed}.png", dpi=140); plt.close(fig)

def plot_comparison(rows: list[dict], output: str | Path) -> None:
  conditions = sorted({r["condition"] for r in rows}); means=[]; stds=[]
  for c in conditions:
    v=np.array([float(r["final_test_accuracy"]) for r in rows if r["condition"]==c]); means.append(v.mean()); stds.append(v.std(ddof=1) if len(v)>1 else 0.)
  fig, ax=plt.subplots(); ax.errorbar(conditions, means, yerr=stds, fmt="o"); ax.set_ylabel("final test accuracy"); fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)

def plot_p_summary(csv_paths: list[tuple[int, Path]], output: str | Path) -> None:
  fig, ax = plt.subplots()
  for seed, path in csv_paths:
    rows = list(csv.DictReader(path.open()))
    ax.plot([int(row["step"]) for row in rows],
            [float(row["p_median"]) for row in rows], label=f"seed {seed}")
  ax.set(xlabel="optimizer step", ylabel="median p_t")
  ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)

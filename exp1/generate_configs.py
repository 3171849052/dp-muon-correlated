#!/usr/bin/env python3
"""Generate and validate the deterministic Experiment 1 CIFAR-10 grid."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXP1 = ROOT / "exp1"
GRID_PATH = EXP1 / "grid.yaml"
CONFIG_ROOT = EXP1 / "config"
MANIFEST_PATH = EXP1 / "manifest.tsv"
ALGORITHMS = ("bandinv", "dpsgd", "dpmuon")
GPU_OFFSETS = {"bandinv": 0, "dpsgd": 1, "dpmuon": 2}

# Allow `python exp1/generate_configs.py` without requiring an editable install.
sys.path.insert(0, str(ROOT / "src"))

from dp_muon.training.cifar10_dpmuon_experiment import load_cifar10_dpmuon_config
from dp_muon.training.cifar10_dpsgd_experiment import (
    load_cifar10_dpsgd_momentum_config,
)
from dp_muon.training.cifar10_experiment import load_cifar10_nonamplified_config


Loader = Callable[[str | Path], object]
LOADERS: dict[str, Loader] = {
    "bandinv": load_cifar10_nonamplified_config,
    "dpsgd": load_cifar10_dpsgd_momentum_config,
    "dpmuon": load_cifar10_dpmuon_config,
}


def _read_yaml(path: Path) -> dict[str, Any]:
  try:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read YAML {path}") from error
  if not isinstance(document, dict):
    raise ValueError(f"{path} must contain a mapping")
  return document


def _number(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
    raise ValueError(f"{name} must be a positive number")
  return float(value)


def _label(value: float) -> str:
  return format(value, ".15g")


def _learning_rate_label(value: float) -> str:
  """Preserves the grid's conventional decimal spelling (including ``1.0``)."""
  return str(value)


def _grid() -> tuple[str, int, list[float], Mapping[str, Any]]:
  grid = _read_yaml(GRID_PATH)
  if set(grid) != {"experiment", "seed", "clip_norms", "algorithms"}:
    raise ValueError("exp1/grid.yaml must contain exactly experiment, seed, clip_norms, algorithms")
  experiment = grid["experiment"]
  seed = grid["seed"]
  clip_norms = grid["clip_norms"]
  algorithms = grid["algorithms"]
  if not isinstance(experiment, str) or not experiment:
    raise ValueError("grid.experiment must be a non-empty string")
  if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
    raise ValueError("grid.seed must be a non-negative integer")
  if not isinstance(clip_norms, list) or len(clip_norms) != 3:
    raise ValueError("grid.clip_norms must contain exactly three values")
  if not isinstance(algorithms, Mapping) or set(algorithms) != set(ALGORITHMS):
    raise ValueError("grid.algorithms must contain exactly bandinv, dpsgd, dpmuon")
  return experiment, seed, [_number(value, "grid.clip_norms") for value in clip_norms], algorithms


def _settings(algorithm: str, entry: Mapping[str, Any]) -> list[tuple[str, dict[str, float]]]:
  if not isinstance(entry.get("template"), str):
    raise ValueError(f"grid.algorithms.{algorithm}.template must be a string")
  if algorithm in {"bandinv", "dpsgd"}:
    if set(entry) != {"template", "learning_rates"}:
      raise ValueError(f"grid.algorithms.{algorithm} must contain template and learning_rates")
    rates = entry["learning_rates"]
    if not isinstance(rates, list) or len(rates) != 3:
      raise ValueError(f"grid.algorithms.{algorithm}.learning_rates must contain exactly three values")
    return [
        (
            f"lr{_learning_rate_label(_number(rate, f'grid.algorithms.{algorithm}.learning_rates'))}",
            {"learning_rate": float(rate)},
        )
        for rate in rates
    ]
  if set(entry) != {"template", "lr_settings"}:
    raise ValueError("grid.algorithms.dpmuon must contain template and lr_settings")
  settings = entry["lr_settings"]
  if not isinstance(settings, list) or len(settings) != 3:
    raise ValueError("grid.algorithms.dpmuon.lr_settings must contain exactly three settings")
  result: list[tuple[str, dict[str, float]]] = []
  for setting in settings:
    if not isinstance(setting, Mapping) or set(setting) != {"name", "muon", "adamw"}:
      raise ValueError("each DP-Muon lr setting must contain name, muon, adamw")
    name = setting["name"]
    if not isinstance(name, str) or not name:
      raise ValueError("DP-Muon lr setting name must be a non-empty string")
    result.append((name, {
        "muon": _number(setting["muon"], "DP-Muon Muon learning rate"),
        "adamw": _number(setting["adamw"], "DP-Muon AdamW learning rate"),
    }))
  if len({name for name, _ in result}) != len(result):
    raise ValueError("DP-Muon lr setting names must be unique")
  return result


def _clean_generated_configs() -> None:
  CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
  for path in CONFIG_ROOT.glob("*.yaml"):
    path.unlink()
  for algorithm in ALGORITHMS:
    directory = CONFIG_ROOT / algorithm
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.yaml"):
      path.unlink()


def generate() -> list[dict[str, str]]:
  """Generate all grid YAML files, validate them, and write the manifest."""
  experiment, seed, clip_norms, algorithms = _grid()
  baselines: dict[str, dict[str, Any]] = {}
  settings: dict[str, list[tuple[str, dict[str, float]]]] = {}
  for algorithm in ALGORITHMS:
    entry = algorithms[algorithm]
    if not isinstance(entry, Mapping):
      raise ValueError(f"grid.algorithms.{algorithm} must be a mapping")
    template = ROOT / entry["template"]
    baseline = _read_yaml(template)
    if baseline.get("algorithm") != algorithm:
      raise ValueError(f"{template} must declare algorithm: {algorithm}")
    LOADERS[algorithm](template)
    baselines[algorithm] = baseline
    settings[algorithm] = _settings(algorithm, entry)

  _clean_generated_configs()
  records: list[dict[str, str]] = []
  for algorithm in ALGORITHMS:
    for setting_index, (lr_setting, rates) in enumerate(settings[algorithm]):
      gpu = (GPU_OFFSETS[algorithm] + setting_index) % 3
      for clip_norm in clip_norms:
        clip_label = _label(clip_norm)
        experiment_id = f"{algorithm}_{lr_setting}_clip{clip_label}"
        document = deepcopy(baselines[algorithm])
        document["algorithm"] = algorithm
        document["experiment"]["name"] = f"{experiment}_{experiment_id}"
        document["experiment"]["seed"] = seed
        document["runtime"]["gpu"] = gpu
        document["training"]["clip_norm"] = clip_norm
        if algorithm == "dpmuon":
          document["muon"]["learning_rate"] = rates["muon"]
          document["adamw"]["learning_rate"] = rates["adamw"]
        else:
          document["training"]["learning_rate"] = rates["learning_rate"]
        if algorithm == "bandinv":
          document["output"]["strategy_dir"] = "artifacts/strategies"
          document["strategy"]["force_refit"] = False
        document["output"]["log_dir"] = f"exp1/logs/{algorithm}"
        relative_config = Path("exp1") / "config" / algorithm / f"{lr_setting}_clip{clip_label}.yaml"
        config_path = ROOT / relative_config
        config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        LOADERS[algorithm](config_path)
        records.append({
            "id": experiment_id,
            "algorithm": algorithm,
            "lr_setting": lr_setting,
            "clip_norm": str(clip_norm),
            "gpu": str(gpu),
            "config": relative_config.as_posix(),
        })

  if len(records) != 27:
    raise AssertionError(f"expected 27 generated configurations, got {len(records)}")
  lines = ["id\talgorithm\tlr_setting\tclip_norm\tgpu\tconfig"]
  lines.extend("\t".join(record[field] for field in lines[0].split("\t")) for record in records)
  MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return records


if __name__ == "__main__":
  generated = generate()
  print(f"generated {len(generated)} configs and {MANIFEST_PATH.relative_to(ROOT)}")

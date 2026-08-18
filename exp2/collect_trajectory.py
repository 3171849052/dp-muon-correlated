#!/usr/bin/env python3
"""Collect clean clipped AdamW gradients before the optimizer update."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import numpy as np

from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny, load_pretrained_vit_tiny
from dp_muon.privacy import make_clipped_gradient_query
from dp_muon.training.cifar10_dpadamw_experiment import load_cifar10_dpadamw_config
from dp_muon.training.cifar10_driver import build_fixed_cycle_logical_schedule, cross_entropy_loss
from dp_muon.training.nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer


def _leaf_at_path(tree: object, parameter_name: str):
  current = tree
  for component in parameter_name.split("/"):
    if isinstance(current, dict):
      current = current[component]
    elif isinstance(current, tuple):
      current = current[int(component)]
    else:
      raise KeyError(f"{parameter_name!r} does not identify a parameter leaf")
  return current


def collect_trajectory(
    *, config_path: str | Path, parameter_name: str, steps: int = 64,
    start_step: int = 0, output: str | Path,
) -> Path:
  """Runs a clean AdamW path and records its post-clipping gradient leaf."""
  if steps < 1:
    raise ValueError("steps must be positive")
  if start_step != 0:
    raise ValueError("Exp2 currently requires start_step == 0")
  config = load_cifar10_dpadamw_config(config_path)
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  if steps > (config.epochs * len(train_images)) // config.logical_batch_size:
    raise ValueError("requested clean window exceeds the configured fixed-cycle run")
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images), batch_size=config.logical_batch_size,
      horizon=steps, min_sep=len(train_images) // config.logical_batch_size,
      max_participations=config.epochs, seed=config.seed,
  )
  model = ViTTiny()
  params = load_pretrained_vit_tiny(
      config.pretrained, key=jax.random.split(jax.random.key(config.seed))[0]
  )
  selected = np.asarray(_leaf_at_path(params, parameter_name))
  if selected.ndim != 2:
    raise ValueError("parameter_name must select a rank-two AdamW parameter")
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2,
      eps=config.eps, weight_decay=config.weight_decay,
  )
  clipped_gradient = make_clipped_gradient_query(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      clip_norm=config.clip_norm, normalize_by=float(config.logical_batch_size),
      batch_argnums=1, keep_batch_dim=True, microbatch_size=config.microbatch_size,
  )
  optimizer_state = optimizer.init(params)
  @jax.jit
  def clean_step(parameters, state, batch):
    gradient = clipped_gradient(parameters, batch)
    updates, new_state = optimizer.update(gradient, state, parameters)
    new_parameters = jax.tree_util.tree_map(
        lambda value, update: value + update, parameters, updates
    )
    return gradient, new_parameters, new_state

  collected: list[np.ndarray] = []
  for _, batch in enumerate(iter_logical_batches(train_images, train_labels, schedule)):
    batch = jax.tree_util.tree_map(jax.numpy.asarray, batch)
    gradient, params, optimizer_state = clean_step(params, optimizer_state, batch)
    collected.append(np.asarray(_leaf_at_path(gradient, parameter_name), dtype=np.float32))
  destination = Path(output)
  destination.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
      destination, g=np.stack(collected), parameter_name=np.asarray(parameter_name),
      start_step=np.asarray(start_step, dtype=np.int64),
      learning_rate=np.asarray(config.learning_rate, dtype=np.float64),
      beta1=np.asarray(config.beta1, dtype=np.float64), beta2=np.asarray(config.beta2, dtype=np.float64),
      eps=np.asarray(config.eps, dtype=np.float64), weight_decay=np.asarray(config.weight_decay, dtype=np.float64),
      clip_norm=np.asarray(config.clip_norm, dtype=np.float64),
  )
  return destination


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=ROOT / "config/cifar10_dpadamw.yaml")
  parser.add_argument("--parameter-name", default="blocks/0/attention/query/kernel")
  parser.add_argument("--steps", type=int, default=64)
  parser.add_argument("--start-step", type=int, default=0)
  parser.add_argument("--output", type=Path, default=ROOT / "exp2/trajectory.npz")
  args = parser.parse_args()
  result = collect_trajectory(
      config_path=args.config, parameter_name=args.parameter_name, steps=args.steps,
      start_step=args.start_step, output=args.output)
  print(f"saved clean AdamW gradient trajectory: {result}")


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Collect a clean, real Muon post-Nesterov/pre-Q matrix trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny, load_pretrained_vit_tiny
from dp_muon.optim import (
    MUON,
    init_muon_nesterov_state,
    muon_nesterov_step,
    vit_muon_parameter_labels,
)
from dp_muon.privacy import make_clipped_gradient_query
from dp_muon.training.cifar10_dpmuon_experiment import load_cifar10_dpmuon_config
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule,
    cross_entropy_loss,
)
from dp_muon.training.nonamplified_dpmuon import make_nonamplified_dpmuon_optimizer


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
    *,
    config_path: str | Path,
    parameter_name: str,
    steps: int = 64,
    start_step: int = 0,
    output: str | Path,
) -> Path:
  """Runs clean clipped Muon and writes consecutive selected pre-Q matrices.

  The optimization path is the existing CIFAR-10 Muon path with its Gaussian
  mechanism omitted.  Clipping remains, but no IID or correlated DP noise is
  sampled.  ``U_t`` is independently observed with the repository's exact
  classic Nesterov recurrence immediately before the Muon Q transform.
  """
  if steps < 1 or start_step < 0:
    raise ValueError("steps must be positive and start_step must be non-negative")
  config = load_cifar10_dpmuon_config(config_path)
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  required_horizon = start_step + steps
  if required_horizon > (config.epochs * len(train_images)) // config.logical_batch_size:
    raise ValueError("requested clean window exceeds the configured fixed-cycle run")
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images),
      batch_size=config.logical_batch_size,
      horizon=required_horizon,
      min_sep=len(train_images) // config.logical_batch_size,
      max_participations=config.epochs,
      seed=config.seed,
  )
  model = ViTTiny()
  parameter_key = jax.random.split(jax.random.key(config.seed))[0]
  params = load_pretrained_vit_tiny(config.pretrained, key=parameter_key)
  selected = jnp.asarray(_leaf_at_path(params, parameter_name))
  selected_label = _leaf_at_path(vit_muon_parameter_labels(params), parameter_name)
  if selected.ndim != 2 or selected_label != MUON:
    raise ValueError("parameter_name must select a rank-two Muon parameter")

  optimizer = make_nonamplified_dpmuon_optimizer(
      muon_learning_rate=config.muon_learning_rate,
      muon_weight_decay=config.muon_weight_decay,
      momentum=config.momentum,
      ns_steps=config.ns_steps,
      consistent_rms=config.consistent_rms,
      adamw_learning_rate=config.adamw_learning_rate,
      adamw_beta1=config.adamw_beta1,
      adamw_beta2=config.adamw_beta2,
      adamw_eps=config.adamw_eps,
      adamw_weight_decay=config.adamw_weight_decay,
      use_bf16_ns=config.use_bf16_ns,
  )
  clipped_gradient = make_clipped_gradient_query(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      clip_norm=config.clip_norm,
      normalize_by=float(config.logical_batch_size),
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=config.microbatch_size,
  )
  optimizer_state = optimizer.init(params)
  nesterov_state = init_muon_nesterov_state(selected)
  collected: list[np.ndarray] = []
  for step, batch in enumerate(iter_logical_batches(train_images, train_labels, schedule)):
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    gradient = clipped_gradient(params, batch)
    selected_gradient = jnp.asarray(_leaf_at_path(gradient, parameter_name))
    pre_q, nesterov_state = muon_nesterov_step(
        nesterov_state, selected_gradient, config.momentum
    )
    if step >= start_step:
      collected.append(np.asarray(pre_q, dtype=np.float32))
    updates, optimizer_state = optimizer.update(gradient, optimizer_state, params)
    params = jax.tree_util.tree_map(lambda value, update: value + update, params, updates)

  destination = Path(output)
  destination.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
      destination,
      u=np.stack(collected),
      learning_rates=np.full(steps, config.muon_learning_rate, dtype=np.float64),
      parameter_name=np.asarray(parameter_name),
      start_step=np.asarray(start_step, dtype=np.int64),
      momentum=np.asarray(config.momentum, dtype=np.float64),
      ns_steps=np.asarray(config.ns_steps, dtype=np.int64),
      consistent_rms=np.asarray(config.consistent_rms, dtype=np.float64),
      use_bf16_ns=np.asarray(config.use_bf16_ns),
  )
  return destination


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=ROOT / "config/cifar10_dpmuon.yaml")
  parser.add_argument(
      "--parameter-name", default="blocks/0/attention/query/kernel",
      help="slash-separated dictionary/tuple path for one rank-two Muon leaf",
  )
  parser.add_argument("--steps", type=int, default=64)
  parser.add_argument("--start-step", type=int, default=0)
  parser.add_argument("--output", type=Path, default=ROOT / "exp1/trajectory.npz")
  args = parser.parse_args()
  result = collect_trajectory(
      config_path=args.config,
      parameter_name=args.parameter_name,
      steps=args.steps,
      start_step=args.start_step,
      output=args.output,
  )
  print(f"saved clean Muon pre-Q trajectory: {result}")


if __name__ == "__main__":
  main()

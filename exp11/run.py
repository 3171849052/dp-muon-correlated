#!/usr/bin/env python3
"""Compare real clean and IID DP-Muon pre-Q spectra at three training steps.

The clean control uses the same per-example clipped query as IID DP-Muon and
simply skips the Gaussian mechanism.  Both controls consume one shared batch
iterator, start from one loaded pretrained snapshot, and use the existing
partitioned DP-Muon trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.optim import extract_pre_q_singular_values
from dp_muon.privacy import calibrate_nonamplified_iid
from dp_muon.training.cifar10_dpmuon_experiment import (
    load_cifar10_dpmuon_config,
)
from dp_muon.training.cifar10_experiment import derive_fixed_cycle_participation
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule,
    cross_entropy_loss,
)
from dp_muon.training.nonamplified_dpmuon import (
    init_nonamplified_dpmuon_state,
    make_nonamplified_dpmuon_train_step,
)
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot

from exp11.plotting import plot_singular_spectra, save_spectra


PARAMETER_PATH = ("blocks", 0, "attention", "query", "kernel")
PARAMETER_NAME = "/".join(str(item) for item in PARAMETER_PATH)
REQUIRED_STEPS = (32, 244, 480)


@dataclass(frozen=True)
class DPMuonSettings:
  """Optimizer settings shared byte-for-byte by both trajectories."""

  muon_learning_rate: float
  muon_weight_decay: float
  momentum: float
  ns_steps: int
  consistent_rms: float
  adamw_learning_rate: float
  adamw_beta1: float
  adamw_beta2: float
  adamw_eps: float
  adamw_weight_decay: float
  microbatch_size: int | None
  use_bf16_ns: bool


@dataclass(frozen=True)
class SpectrumResult:
  steps: np.ndarray
  clean_singular_values: np.ndarray
  dp_singular_values: np.ndarray


def _validate_steps(steps: Iterable[int], horizon: int) -> tuple[int, ...]:
  requested = tuple(int(step) for step in steps)
  if not requested or any(step < 1 or step > horizon for step in requested):
    raise ValueError("requested spectrum steps must lie in [1, horizon]")
  if tuple(sorted(set(requested))) != requested:
    raise ValueError("requested spectrum steps must be strictly increasing")
  return requested


def run_paired_trajectories(
    *,
    initial_params: Any,
    batches: Iterable[Any],
    horizon: int,
    calibration: Any,
    loss_fn: Callable[..., Any],
    settings: DPMuonSettings,
    seed: int,
    steps: Iterable[int],
) -> SpectrumResult:
  """Run two paired states and collect only the selected pre-Q spectra."""
  requested = _validate_steps(steps, horizon)
  clean_step, clean_optimizer = make_nonamplified_dpmuon_train_step(
      loss_fn,
      calibration,
      muon_learning_rate=settings.muon_learning_rate,
      muon_weight_decay=settings.muon_weight_decay,
      momentum=settings.momentum,
      ns_steps=settings.ns_steps,
      consistent_rms=settings.consistent_rms,
      adamw_learning_rate=settings.adamw_learning_rate,
      adamw_beta1=settings.adamw_beta1,
      adamw_beta2=settings.adamw_beta2,
      adamw_eps=settings.adamw_eps,
      adamw_weight_decay=settings.adamw_weight_decay,
      microbatch_size=settings.microbatch_size,
      use_bf16_ns=settings.use_bf16_ns,
      add_noise=False,
      pre_q_parameter_path=PARAMETER_PATH,
  )
  dp_step, dp_optimizer = make_nonamplified_dpmuon_train_step(
      loss_fn,
      calibration,
      muon_learning_rate=settings.muon_learning_rate,
      muon_weight_decay=settings.muon_weight_decay,
      momentum=settings.momentum,
      ns_steps=settings.ns_steps,
      consistent_rms=settings.consistent_rms,
      adamw_learning_rate=settings.adamw_learning_rate,
      adamw_beta1=settings.adamw_beta1,
      adamw_beta2=settings.adamw_beta2,
      adamw_eps=settings.adamw_eps,
      adamw_weight_decay=settings.adamw_weight_decay,
      microbatch_size=settings.microbatch_size,
      use_bf16_ns=settings.use_bf16_ns,
      add_noise=True,
      pre_q_parameter_path=PARAMETER_PATH,
  )
  # The immutable parameter tree and the two optimizer states are initialized
  # from exactly the same values.  Only the DP state subsequently consumes its
  # key to sample IID Gaussian noise.
  _, noise_key = jax.random.split(jax.random.key(seed))
  clean_state = init_nonamplified_dpmuon_state(
      initial_params, noise_key, clean_optimizer
  )
  dp_state = init_nonamplified_dpmuon_state(
      initial_params, noise_key, dp_optimizer
  )
  compiled_clean_step = jax.jit(clean_step)
  compiled_dp_step = jax.jit(dp_step)
  clean_spectra: list[np.ndarray] = []
  dp_spectra: list[np.ndarray] = []
  requested_set = set(requested)
  batch_iterator = iter(batches)
  for step in range(1, horizon + 1):
    try:
      batch = next(batch_iterator)
    except StopIteration as error:
      raise ValueError("batches must contain exactly horizon batches") from error
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    # Both compiled updates see this same batch object before the iterator is
    # advanced.  Their states remain independent after the first update.
    clean_state = compiled_clean_step(clean_state, batch)
    dp_state = compiled_dp_step(dp_state, batch)
    if step in requested_set:
      clean_spectra.append(
          np.asarray(extract_pre_q_singular_values(clean_state.optimizer_state),
                     dtype=np.float64).copy()
      )
      dp_spectra.append(
          np.asarray(extract_pre_q_singular_values(dp_state.optimizer_state),
                     dtype=np.float64).copy()
      )
  try:
    next(batch_iterator)
  except StopIteration:
    return SpectrumResult(
        steps=np.asarray(requested, dtype=np.int32),
        clean_singular_values=np.stack(clean_spectra),
        dp_singular_values=np.stack(dp_spectra),
    )
  raise ValueError("batches must contain exactly horizon batches")


def _settings_from_config(config: Any) -> DPMuonSettings:
  return DPMuonSettings(
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
      microbatch_size=config.microbatch_size,
      use_bf16_ns=config.use_bf16_ns,
  )


def run_formal(
    config_path: str | Path = "config/cifar10_dpmuon.yaml",
    output_dir: str | Path = "exp11/results",
    *,
    steps: Iterable[int] = REQUIRED_STEPS,
) -> SpectrumResult:
  """Run the full config-defined fixed-cycle horizon and write both artifacts."""
  config = load_cifar10_dpmuon_config(config_path)
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  participation = derive_fixed_cycle_participation(
      len(train_images), config.epochs, config.logical_batch_size
  )
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images),
      batch_size=config.logical_batch_size,
      horizon=participation.horizon,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      seed=config.seed,
  )
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  del noise_key
  snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  calibration = calibrate_nonamplified_iid(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.logical_batch_size),
      adjacency=config.adjacency,
      max_participations=participation.max_participations,
  )
  model = ViTTiny()
  result = run_paired_trajectories(
      initial_params=snapshot.params,
      batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=participation.horizon,
      calibration=calibration,
      loss_fn=lambda params, batch: cross_entropy_loss(params, batch, model),
      settings=_settings_from_config(config),
      seed=config.seed,
      steps=steps,
  )
  output_dir = Path(output_dir)
  spectra_path = save_spectra(
      output_dir / "spectra.npz",
      steps=result.steps,
      parameter_name=PARAMETER_NAME,
      clean_singular_values=result.clean_singular_values,
      dp_singular_values=result.dp_singular_values,
  )
  plot_singular_spectra(spectra_path, output_dir / "singular_spectra.png")
  print(f"wrote {spectra_path}")
  print(f"wrote {output_dir / 'singular_spectra.png'}")
  return result


def _smoke_params() -> dict[str, Any]:
  matrix = lambda: {"kernel": jnp.eye(3, dtype=jnp.float32), "bias": jnp.zeros((3,), jnp.float32)}
  return {
      "blocks": ({
          "attention": {name: matrix() for name in ("query", "key", "value", "out")},
          "mlp": {name: matrix() for name in ("dense0", "dense1")},
      },),
      "head": matrix(),
  }


def run_smoke(output_dir: str | Path) -> SpectrumResult:
  """Run three tiny paired updates without CIFAR or a pretrained artifact."""
  params = _smoke_params()

  def loss_fn(parameters, batch):
    return (jnp.sum(parameters["blocks"][0]["attention"]["query"]["kernel"])
            + jnp.sum(parameters["head"]["kernel"])) * batch["scale"][0]

  calibration = calibrate_nonamplified_iid(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=0.5,
      normalize_by=2.0,
      adjacency="add_remove",
      max_participations=2,
  )
  calibration = replace(calibration, iid_noise_std=0.15)
  settings = DPMuonSettings(
      muon_learning_rate=0.01,
      muon_weight_decay=0.0,
      momentum=0.8,
      ns_steps=2,
      consistent_rms=0.2,
      adamw_learning_rate=0.01,
      adamw_beta1=0.9,
      adamw_beta2=0.99,
      adamw_eps=1e-6,
      adamw_weight_decay=0.0,
      microbatch_size=None,
      use_bf16_ns=False,
  )
  batches = [{"scale": jnp.asarray([value, -value], jnp.float32)}
             for value in (1.0, 0.5, 2.0)]
  result = run_paired_trajectories(
      initial_params=params,
      batches=batches,
      horizon=3,
      calibration=calibration,
      loss_fn=loss_fn,
      settings=settings,
      seed=7,
      steps=(1, 2, 3),
  )
  output_dir = Path(output_dir)
  spectra_path = save_spectra(
      output_dir / "spectra.npz",
      steps=result.steps,
      parameter_name=PARAMETER_NAME,
      clean_singular_values=result.clean_singular_values,
      dp_singular_values=result.dp_singular_values,
  )
  plot_singular_spectra(spectra_path, output_dir / "singular_spectra.png")
  print(f"wrote {spectra_path}")
  print(f"wrote {output_dir / 'singular_spectra.png'}")
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="config/cifar10_dpmuon.yaml")
  parser.add_argument("--output-dir", default="exp11/results")
  parser.add_argument("--steps", nargs=3, type=int, default=list(REQUIRED_STEPS),
                      metavar=("EARLY", "MIDDLE", "LATE"))
  parser.add_argument("--smoke", action="store_true",
                      help="run a tiny synthetic end-to-end check")
  args = parser.parse_args()
  if args.smoke:
    run_smoke(args.output_dir)
  else:
    run_formal(args.config, args.output_dir, steps=args.steps)


if __name__ == "__main__":
  main()

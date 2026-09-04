#!/usr/bin/env python3
"""Run the paired, cross-layer Exp11b IID DP-Muon spectrum experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.optim import PreQSVDState, extract_pre_q_singular_values
from dp_muon.privacy import PrivacyCalibration, calibrate_nonamplified_iid
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

from exp11b.plotting import (
    plot_singular_spectra,
    save_spectra,
    save_spectra_csv,
)


TARGET_BLOCKS = (0, 5, 11)
TARGET_PARAMETER_PATHS = tuple(
    ("blocks", block, "attention", "query", "kernel")
    for block in TARGET_BLOCKS
)
TARGET_LAYER_NAMES = tuple("/".join(str(item) for item in path)
                           for path in TARGET_PARAMETER_PATHS)
TARGET_EPSILONS = (3, 8)
REQUIRED_STEPS = (32, 244, 480)


@dataclass(frozen=True)
class DPMuonSettings:
  """Muon/AdamW settings shared by all three trajectories."""

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
class PairedSpectrumResult:
  """One clean/DP pair, with dimensions [step, layer, singular-index]."""

  steps: np.ndarray
  layers: tuple[str, ...]
  clean_singular_values: np.ndarray
  dp_singular_values: np.ndarray


@dataclass(frozen=True)
class SpectrumResult:
  """The complete result, with dimensions [epsilon, step, layer, index]."""

  epsilons: np.ndarray
  steps: np.ndarray
  layers: tuple[str, ...]
  clean_singular_values: np.ndarray
  dp_singular_values: np.ndarray


def _validate_steps(steps: Iterable[int], horizon: int) -> tuple[int, ...]:
  requested = tuple(int(step) for step in steps)
  if not requested or any(step < 1 or step > horizon for step in requested):
    raise ValueError("requested spectrum steps must lie in [1, horizon]")
  if tuple(sorted(set(requested))) != requested:
    raise ValueError("requested spectrum steps must be strictly increasing")
  return requested


def _make_train_step(
    loss_fn: Callable[..., Any],
    calibration: PrivacyCalibration,
    settings: DPMuonSettings,
    *,
    add_noise: bool,
):
  """Build one existing IID DP-Muon trainer with the multi-layer hook."""
  return make_nonamplified_dpmuon_train_step(
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
      add_noise=add_noise,
      pre_q_parameter_paths=TARGET_PARAMETER_PATHS,
  )


def _extract_target_spectra(optimizer_state: Any) -> np.ndarray:
  """Extract the three existing hook states without retaining any matrix."""
  values: list[np.ndarray] = []

  def visit(node: Any) -> None:
    if isinstance(node, PreQSVDState):
      values.append(np.asarray(
          extract_pre_q_singular_values(node), dtype=np.float64
      ).copy())
      return
    if isinstance(node, Mapping):
      for child in node.values():
        visit(child)
      return
    if isinstance(node, tuple):
      for child in node:
        visit(child)

  visit(optimizer_state)
  if len(values) != len(TARGET_PARAMETER_PATHS):
    raise ValueError(
        "optimizer state must contain one pre-Q SVD state per target layer"
    )
  dimensions = {value.shape for value in values}
  if len(dimensions) != 1:
    raise ValueError("target layers must have the same matrix shape")
  spectra = np.stack(values)
  if np.any(np.diff(spectra, axis=-1) > 0):
    raise ValueError("pre-Q singular values must be descending")
  return spectra


def _run_trajectory_group(
    *,
    initial_params: Any,
    batches: Iterable[Any],
    horizon: int,
    calibrations: Mapping[int, PrivacyCalibration],
    loss_fn: Callable[..., Any],
    settings: DPMuonSettings,
    seed: int,
    steps: Iterable[int],
) -> SpectrumResult:
  """Run clean, epsilon=3, and epsilon=8 on one shared batch iterator."""
  requested = _validate_steps(steps, horizon)
  for epsilon in TARGET_EPSILONS:
    if epsilon not in calibrations:
      raise ValueError(f"missing calibration for epsilon={epsilon}")

  clean_step, clean_optimizer = _make_train_step(
      loss_fn, calibrations[TARGET_EPSILONS[0]], settings, add_noise=False
  )
  dp_steps: list[Any] = []
  dp_optimizers: list[Any] = []
  for epsilon in TARGET_EPSILONS:
    step, optimizer = _make_train_step(
        loss_fn, calibrations[epsilon], settings, add_noise=True
    )
    dp_steps.append(step)
    dp_optimizers.append(optimizer)

  # All states receive the same immutable parameter tree and the same RNG key.
  # Clean never consumes its key; epsilon 3 and epsilon 8 therefore use the
  # same IID standard-normal realization with different calibrated scales.
  _, shared_noise_key = jax.random.split(jax.random.key(seed))
  states = [
      init_nonamplified_dpmuon_state(initial_params, shared_noise_key, clean_optimizer),
      *[
          init_nonamplified_dpmuon_state(
              initial_params, shared_noise_key, optimizer
          )
          for optimizer in dp_optimizers
      ],
  ]
  train_steps = [jax.jit(clean_step), *(jax.jit(step) for step in dp_steps)]

  collected: list[list[np.ndarray]] = [[], [], []]
  requested_set = set(requested)
  batch_iterator = iter(batches)
  for step_number in range(1, horizon + 1):
    try:
      batch = next(batch_iterator)
    except StopIteration as error:
      raise ValueError("batches must contain exactly horizon batches") from error
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    # The same batch object is passed to each closed-loop state before the
    # iterator advances.  Each update uses that trajectory's real optimizer
    # state, including its Muon momentum/Nesterov state.
    states = [
        train_step(state, batch)
        for train_step, state in zip(train_steps, states, strict=True)
    ]
    if step_number in requested_set:
      for index, state in enumerate(states):
        collected[index].append(_extract_target_spectra(state.optimizer_state))

  try:
    next(batch_iterator)
  except StopIteration:
    pass
  else:
    raise ValueError("batches must contain exactly horizon batches")

  clean = np.stack(collected[0])
  dp = np.stack([np.stack(collected[index]) for index in (1, 2)])
  clean_repeated = np.stack([clean, clean.copy()])
  return SpectrumResult(
      epsilons=np.asarray(TARGET_EPSILONS, dtype=np.int32),
      steps=np.asarray(requested, dtype=np.int32),
      layers=TARGET_LAYER_NAMES,
      clean_singular_values=clean_repeated,
      dp_singular_values=dp,
  )


def run_paired_trajectories(
    *,
    initial_params: Any,
    batches: Iterable[Any],
    horizon: int,
    calibration: PrivacyCalibration,
    loss_fn: Callable[..., Any],
    settings: DPMuonSettings,
    seed: int,
    steps: Iterable[int],
) -> PairedSpectrumResult:
  """Run one clean/DP pair; retained as a small unit-testable Exp11 API."""
  result = _run_trajectory_group(
      initial_params=initial_params,
      batches=batches,
      horizon=horizon,
      calibrations={3: calibration, 8: calibration},
      loss_fn=loss_fn,
      settings=settings,
      seed=seed,
      steps=steps,
  )
  return PairedSpectrumResult(
      steps=result.steps,
      layers=result.layers,
      clean_singular_values=result.clean_singular_values[0],
      dp_singular_values=result.dp_singular_values[0],
  )


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


def _calibrations(config: Any, max_participations: int) -> dict[int, PrivacyCalibration]:
  if not np.isclose(config.clip_norm, 1.0, rtol=0.0, atol=0.0):
    raise ValueError("Exp11b requires clip_norm=1.0")
  if not np.isclose(config.delta, 1e-5, rtol=0.0, atol=0.0):
    raise ValueError("Exp11b requires delta=1e-5")
  result = {
      epsilon: calibrate_nonamplified_iid(
          epsilon=epsilon,
          delta=1e-5,
          clip_norm=1.0,
          normalize_by=float(config.logical_batch_size),
          adjacency=config.adjacency,
          max_participations=max_participations,
      )
      for epsilon in TARGET_EPSILONS
  }
  if not result[8].iid_noise_std < result[3].iid_noise_std:
    raise ValueError("epsilon=8 must have a smaller IID noise standard deviation")
  return result


def _write_results(result: SpectrumResult, output_dir: str | Path) -> None:
  output_dir = Path(output_dir)
  spectra_path = save_spectra(output_dir / "spectra.npz", result=result)
  csv_path = save_spectra_csv(spectra_path, output_dir / "spectra.csv")
  for epsilon in TARGET_EPSILONS:
    plot_singular_spectra(
        spectra_path,
        output_dir / f"singular_spectra_eps{epsilon}.png",
        epsilon=epsilon,
    )
  print(f"wrote {spectra_path}")
  print(f"wrote {csv_path}")
  for epsilon in TARGET_EPSILONS:
    print(f"wrote {output_dir / f'singular_spectra_eps{epsilon}.png'}")


def run_formal(
    config_path: str | Path = "config/cifar10_dpmuon.yaml",
    output_dir: str | Path = "exp11b/results",
    *,
    steps: Iterable[int] = REQUIRED_STEPS,
) -> SpectrumResult:
  """Run the full config-defined fixed-cycle horizon and write Exp11b files."""
  requested_steps = tuple(int(step) for step in steps)
  if requested_steps != REQUIRED_STEPS:
    raise ValueError("formal Exp11b recording steps are exactly 32, 244, and 480")
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
  parameter_key, trajectory_key = jax.random.split(jax.random.key(config.seed))
  snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  del trajectory_key
  calibrations = _calibrations(config, participation.max_participations)
  for epsilon in TARGET_EPSILONS:
    print(
        f"epsilon={epsilon} calibrated_noise_std="
        f"{calibrations[epsilon].iid_noise_std:.17e}",
        flush=True,
    )
  model = ViTTiny()
  result = _run_trajectory_group(
      initial_params=snapshot.params,
      batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=participation.horizon,
      calibrations=calibrations,
      loss_fn=lambda params, batch: cross_entropy_loss(params, batch, model),
      settings=_settings_from_config(config),
      seed=config.seed,
      steps=steps,
  )
  _write_results(result, output_dir)
  return result


def _smoke_params() -> dict[str, Any]:
  def matrix(scale: float) -> dict[str, jax.Array]:
    return {
        "kernel": jnp.eye(3, dtype=jnp.float32) * scale,
        "bias": jnp.zeros((3,), dtype=jnp.float32),
    }

  blocks = []
  for block in range(12):
    blocks.append({
        "attention": {
            name: matrix(1.0 + 0.01 * block)
            for name in ("query", "key", "value", "out")
        },
        "mlp": {name: matrix(1.0) for name in ("dense0", "dense1")},
    })
  return {"blocks": tuple(blocks), "head": matrix(1.0)}


def _smoke_settings() -> DPMuonSettings:
  return DPMuonSettings(
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


def run_smoke(output_dir: str | Path) -> SpectrumResult:
  """Run a tiny 12-block, three-trajectory end-to-end check."""
  params = _smoke_params()

  def loss_fn(parameters: dict, batch: dict) -> jax.Array:
    selected = sum(
        jnp.sum(parameters["blocks"][block]["attention"]["query"]["kernel"])
        for block in TARGET_BLOCKS
    )
    return (selected + jnp.sum(parameters["head"]["kernel"])) * batch["scale"][0]

  calibrations = {
      epsilon: calibrate_nonamplified_iid(
          epsilon=epsilon,
          delta=1e-5,
          clip_norm=1.0,
          normalize_by=2.0,
          adjacency="add_remove",
          max_participations=2,
      )
      for epsilon in TARGET_EPSILONS
  }
  for epsilon in TARGET_EPSILONS:
    print(
        f"epsilon={epsilon} calibrated_noise_std="
        f"{calibrations[epsilon].iid_noise_std:.17e}",
        flush=True,
    )
  batches = [
      {"scale": jnp.asarray([value, value], dtype=jnp.float32)}
      for value in (1.0, 0.5, 2.0)
  ]
  result = _run_trajectory_group(
      initial_params=params,
      batches=batches,
      horizon=3,
      calibrations=calibrations,
      loss_fn=loss_fn,
      settings=_smoke_settings(),
      seed=7,
      steps=(1, 2, 3),
  )
  _write_results(result, output_dir)
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="config/cifar10_dpmuon.yaml")
  parser.add_argument("--output-dir", default="exp11b/results")
  parser.add_argument(
      "--steps", nargs=3, type=int, default=list(REQUIRED_STEPS),
      metavar=("EARLY", "MIDDLE", "LATE"),
  )
  parser.add_argument("--smoke", action="store_true",
                      help="run a tiny synthetic end-to-end check")
  args = parser.parse_args()
  if args.smoke:
    run_smoke(args.output_dir)
  else:
    run_formal(args.config, args.output_dir, steps=args.steps)


if __name__ == "__main__":
  main()

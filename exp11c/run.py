#!/usr/bin/env python3
"""Directly test ideal Muon's scale blindness on the Exp11b trajectories."""

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
from dp_muon.optim import PreQMatrixState, extract_pre_q_matrix
from dp_muon.privacy import PrivacyCalibration
from dp_muon.training.cifar10_dpmuon_experiment import load_cifar10_dpmuon_config
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

from exp11b.run import (
    DPMuonSettings,
    REQUIRED_STEPS,
    TARGET_BLOCKS,
    TARGET_EPSILONS,
    TARGET_LAYER_NAMES,
    TARGET_PARAMETER_PATHS,
    _calibrations,
    _settings_from_config,
)
from exp11c.plotting import (
    PAIR_NAMES,
    TRAJECTORIES,
    plot_scale_blindness,
    save_scale_blindness,
    save_scale_blindness_csv,
)


@dataclass(frozen=True)
class ScaleBlindnessResult:
  """Compact diagnostics with dimensions stated for every array.

  No pre-Q or ideal-Q matrix is retained here.  Matrices exist only while a
  requested checkpoint is being converted into the metrics below.

  ``matrix_frobenius_norms`` has shape ``[trajectory, step, layer]``;
  ``matrix_scale_ratios`` has shape ``[step, layer]``; the two ideal-Q arrays
  have shape ``[step, layer, pair]``.
  """

  epsilons: np.ndarray
  steps: np.ndarray
  layers: tuple[str, ...]
  trajectories: tuple[str, ...]
  pair_names: tuple[str, ...]
  noise_stds: np.ndarray
  matrix_frobenius_norms: np.ndarray
  matrix_scale_ratios: np.ndarray
  ideal_q_pairwise_frobenius_distances: np.ndarray
  ideal_q_pairwise_cosines: np.ndarray


def ideal_muon_q(matrix: Any) -> np.ndarray:
  """Return the exact float64 polar factor used by the Exp11c analysis."""
  raw = np.asarray(matrix)
  if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.floating):
    raise ValueError("ideal Muon Q expects a floating rank-two matrix")
  x = raw.astype(np.float64)
  if not np.all(np.isfinite(x)):
    raise ValueError("ideal Muon Q input must be finite")
  u, _, vh = np.linalg.svd(x.astype(np.float64), full_matrices=False)
  return u @ vh


def _validate_steps(steps: Iterable[int], horizon: int) -> tuple[int, ...]:
  requested = tuple(int(step) for step in steps)
  if not requested or any(step < 1 or step > horizon for step in requested):
    raise ValueError("requested scale-blindness steps must lie in [1, horizon]")
  if tuple(sorted(set(requested))) != requested:
    raise ValueError("requested scale-blindness steps must be strictly increasing")
  return requested


def _extract_target_pre_q_matrices(optimizer_state: Any) -> np.ndarray:
  """Copy the three latest hook matrices to host in target-layer order."""
  values: list[np.ndarray] = []

  def visit(node: Any) -> None:
    if isinstance(node, PreQMatrixState):
      # This is the only host transfer in the trajectory collector.  The
      # caller invokes it only on a requested recording step.
      values.append(np.asarray(extract_pre_q_matrix(node), dtype=np.float64).copy())
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
        "optimizer state must contain three pre-Q matrix states, one per target layer"
    )
  shapes = {value.shape for value in values}
  if len(shapes) != 1 or any(len(shape) != 2 for shape in shapes):
    raise ValueError("target pre-Q matrices must have one common rank-two shape")
  if any(not np.all(np.isfinite(value)) for value in values):
    raise ValueError("target pre-Q matrices must be finite")
  return np.stack(values)


def _pairwise_metrics(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Compute pairwise ideal-Q distances and cosine similarities.

  ``q_values`` has shape [trajectory, row, column].  Pair order is fixed by
  ``PAIR_NAMES`` and is intentionally part of the artifact schema.
  """
  if q_values.ndim != 3 or q_values.shape[0] != len(TRAJECTORIES):
    raise ValueError("ideal-Q values must have shape [3, rows, columns]")
  pairs = ((0, 1), (0, 2), (1, 2))
  distances = []
  cosines = []
  for left, right in pairs:
    left_q, right_q = q_values[left], q_values[right]
    left_norm = float(np.linalg.norm(left_q))
    right_norm = float(np.linalg.norm(right_q))
    distances.append(float(np.linalg.norm(left_q - right_q)))
    if left_norm == 0.0 or right_norm == 0.0:
      cosines.append(1.0 if left_norm == right_norm else 0.0)
    else:
      cosines.append(float(np.sum(left_q * right_q) / (left_norm * right_norm)))
  return np.asarray(distances, dtype=np.float64), np.asarray(cosines, dtype=np.float64)


def _safe_scale_ratio(numerator: float, denominator: float) -> float:
  if denominator > 0.0:
    return float(numerator / denominator)
  return 0.0 if numerator == 0.0 else float(np.finfo(np.float64).max)


def _make_train_step(
    loss_fn: Callable[..., Any],
    calibration: PrivacyCalibration,
    settings: DPMuonSettings,
    *,
    add_noise: bool,
):
  """Build Exp11b's trainer with a matrix-only pre-Q capture hook."""
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
      pre_q_matrix_parameter_paths=TARGET_PARAMETER_PATHS,
  )


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
) -> ScaleBlindnessResult:
  """Run one shared-schedule clean/eps3/eps8 group and reduce target matrices."""
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

  # The DP branches start from one key.  JAX's tree-shaped sampler therefore
  # creates the same standard-normal realization for both; only iid_noise_std
  # differs.  Clean never consumes its copy of the key.
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

  norm_rows: list[np.ndarray] = []
  distance_rows: list[np.ndarray] = []
  cosine_rows: list[np.ndarray] = []
  requested_set = set(requested)
  batch_iterator = iter(batches)
  for step_number in range(1, horizon + 1):
    try:
      batch = next(batch_iterator)
    except StopIteration as error:
      raise ValueError("batches must contain exactly horizon batches") from error
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    states = [
        train_step(state, batch)
        for train_step, state in zip(train_steps, states, strict=True)
    ]
    if step_number in requested_set:
      matrices = np.stack([
          _extract_target_pre_q_matrices(state.optimizer_state)
          for state in states
      ])
      # Matrices are reduced immediately.  Neither this array nor the ideal-Q
      # array is placed in ScaleBlindnessResult or any output artifact.
      norms = np.linalg.norm(matrices, axis=(-2, -1)).astype(np.float64)
      ideal_q = np.stack([
          np.stack([ideal_muon_q(matrix) for matrix in trajectory])
          for trajectory in matrices
      ])
      distances = []
      cosines = []
      for layer_index in range(len(TARGET_PARAMETER_PATHS)):
        layer_distances, layer_cosines = _pairwise_metrics(
            ideal_q[:, layer_index]
        )
        distances.append(layer_distances)
        cosines.append(layer_cosines)
      norm_rows.append(norms)
      distance_rows.append(np.stack(distances))
      cosine_rows.append(np.stack(cosines))

  try:
    next(batch_iterator)
  except StopIteration:
    pass
  else:
    raise ValueError("batches must contain exactly horizon batches")

  # Collection is naturally [step, trajectory, layer]; expose the artifact
  # axis order promised by ScaleBlindnessResult.
  norms = np.transpose(np.stack(norm_rows), (1, 0, 2))
  distances = np.stack(distance_rows)
  cosines = np.stack(cosine_rows)
  ratios = np.zeros((len(requested), len(TARGET_PARAMETER_PATHS)), dtype=np.float64)
  for step_index in range(len(requested)):
    for layer_index in range(len(TARGET_PARAMETER_PATHS)):
      ratios[step_index, layer_index] = _safe_scale_ratio(
          float(norms[1, step_index, layer_index]),
          float(norms[2, step_index, layer_index]),
      )
  noise_stds = np.asarray((
      0.0,
      float(calibrations[3].iid_noise_std),
      float(calibrations[8].iid_noise_std),
  ), dtype=np.float64)
  return ScaleBlindnessResult(
      epsilons=np.asarray(TARGET_EPSILONS, dtype=np.int32),
      steps=np.asarray(requested, dtype=np.int32),
      layers=TARGET_LAYER_NAMES,
      trajectories=TRAJECTORIES,
      pair_names=PAIR_NAMES,
      noise_stds=noise_stds,
      matrix_frobenius_norms=norms,
      matrix_scale_ratios=ratios,
      ideal_q_pairwise_frobenius_distances=distances,
      ideal_q_pairwise_cosines=cosines,
  )


def run_formal(
    config_path: str | Path = "config/cifar10_dpmuon.yaml",
    output_dir: str | Path = "exp11c/results",
    *,
    steps: Iterable[int] = REQUIRED_STEPS,
) -> ScaleBlindnessResult:
  """Run Exp11c on Exp11b's full fixed-cycle CIFAR-10 trajectory."""
  requested_steps = tuple(int(step) for step in steps)
  if requested_steps != REQUIRED_STEPS:
    raise ValueError("formal Exp11c recording steps are exactly 32, 244, and 480")
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
      steps=requested_steps,
  )
  _write_results(result, output_dir)
  return result


def _write_results(result: ScaleBlindnessResult, output_dir: str | Path) -> None:
  output_dir = Path(output_dir)
  metrics_path = save_scale_blindness(output_dir / "scale_blindness.npz", result=result)
  csv_path = save_scale_blindness_csv(metrics_path, output_dir / "scale_blindness.csv")
  plot_path = plot_scale_blindness(metrics_path, output_dir / "scale_blindness.png")
  print(f"wrote {metrics_path}")
  print(f"wrote {csv_path}")
  print(f"wrote {plot_path}")


def _smoke_params() -> dict[str, Any]:
  from exp11b.run import _smoke_params as exp11b_smoke_params

  return exp11b_smoke_params()


def _smoke_settings() -> DPMuonSettings:
  from exp11b.run import _smoke_settings as exp11b_smoke_settings

  return exp11b_smoke_settings()


def run_smoke(output_dir: str | Path) -> ScaleBlindnessResult:
  """Run a tiny three-trajectory matrix-capture and ideal-Q check."""
  params = _smoke_params()

  def loss_fn(parameters: dict, batch: dict) -> jax.Array:
    selected = sum(
        jnp.sum(parameters["blocks"][block]["attention"]["query"]["kernel"])
        for block in TARGET_BLOCKS
    )
    return (selected + jnp.sum(parameters["head"]["kernel"])) * batch["scale"][0]

  from dp_muon.privacy import calibrate_nonamplified_iid

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
  result = _run_trajectory_group(
      initial_params=params,
      batches=[
          {"scale": jnp.asarray([value, value], dtype=jnp.float32)}
          for value in (1.0, 0.5, 2.0)
      ],
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
  parser.add_argument("--output-dir", default="exp11c/results")
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

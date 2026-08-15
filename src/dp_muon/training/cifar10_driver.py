"""Orchestration for CIFAR-10 fine-tuning with the existing M6 trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.bandinvmf import BandInvMFStrategy, load_bandinv_strategy
from dp_muon.data import iter_logical_batches, load_cifar10, prepare_cifar10_batch
from dp_muon.models import ViTTiny, load_pretrained_vit_tiny
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
    certify_participation_schedule,
)

from .checkpoint import load_checkpoint, save_checkpoint
from .nonamplified_linear import (
    NonAmplifiedBandInvState,
    init_nonamplified_bandinv_state,
    make_nonamplified_bandinv_train_step,
)


@dataclass(frozen=True)
class Cifar10TrainConfig:
  strategy: str
  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
  momentum: float
  learning_rate: float
  seed: int
  checkpoint_dir: str
  eval_every: int
  adjacency: str = "add_remove"

  def __post_init__(self) -> None:
    if self.batch_size < 1:
      raise ValueError("batch_size must be positive")
    if self.microbatch_size is not None and self.microbatch_size < 1:
      raise ValueError("microbatch_size must be positive when supplied")
    if (
        self.microbatch_size is not None
        and self.batch_size % self.microbatch_size != 0
    ):
      raise ValueError("batch_size must be divisible by microbatch_size")
    if self.eval_every < 1:
      raise ValueError("eval_every must be positive")


def build_logical_schedule(
    *, num_examples: int, batch_size: int, strategy: BandInvMFStrategy, seed: int
) -> list[np.ndarray]:
  """Builds a deterministic fixed-size schedule and certifies it against M4."""
  if num_examples < 1 or batch_size < 1:
    raise ValueError("num_examples and batch_size must be positive")
  if batch_size > num_examples:
    raise ValueError("batch_size must not exceed number of training examples")
  # Moving through one shuffled cyclic order makes every repeated record at
  # least floor(num_examples / batch_size) steps apart.  Certification below is
  # the authority for the fitted min-separation/max-participation contract.
  permutation = np.random.default_rng(seed).permutation(num_examples)
  schedule = [
      permutation[(step * batch_size + np.arange(batch_size)) % num_examples].astype(np.int32)
      for step in range(strategy.horizon)
  ]
  certify_participation_schedule(
      schedule,
      ParticipationSpec(strategy.horizon, strategy.min_sep, strategy.max_participations),
  )
  return schedule


def cross_entropy_loss(params: dict, batch: dict[str, jax.Array], model: ViTTiny) -> jax.Array:
  logits = model.apply(params, batch["image"])
  labels = batch["label"]
  return -jax.nn.log_softmax(logits)[0, labels[0]]


def evaluate_classifier(
    params: dict, model: ViTTiny, images: np.ndarray, labels: np.ndarray, *, batch_size: int
) -> float:
  correct = 0
  for offset in range(0, len(images), batch_size):
    batch = prepare_cifar10_batch(images[offset : offset + batch_size], labels[offset : offset + batch_size])
    predictions = np.asarray(jnp.argmax(model.apply(params, jnp.asarray(batch["image"])), axis=-1))
    correct += int(np.sum(predictions == batch["label"]))
  return correct / len(images)


def run_training(
    *,
    initial_state: NonAmplifiedBandInvState,
    train_step: Callable[[NonAmplifiedBandInvState, Any], NonAmplifiedBandInvState],
    logical_batches: Iterable[Any],
    horizon: int,
    experiment_config: dict[str, Any],
    artifact_identifiers: dict[str, str],
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    eval_every: int = 1,
    evaluate: Callable[[NonAmplifiedBandInvState], float] | None = None,
) -> tuple[NonAmplifiedBandInvState, list[dict[str, float | int]]]:
  """Runs exactly one M6 call for each logical batch, with optional resume.

  The iterable is deliberately consumed one batch at a time.  No accumulation,
  no splitting, and no retry/skip path can introduce another DP query.
  """
  if horizon < 1 or eval_every < 1:
    raise ValueError("horizon and eval_every must be positive")
  state, start = initial_state, 0
  if resume_checkpoint is not None:
    saved = load_checkpoint(resume_checkpoint)
    if saved["artifact_identifiers"] != dict(artifact_identifiers):
      raise ValueError("checkpoint artifact identifiers do not match this run")
    if saved["experiment_config"] != dict(experiment_config):
      raise ValueError("checkpoint experiment config does not match this run")
    state, start = saved["state"], int(saved["current_step"])
    if start > horizon:
      raise ValueError("checkpoint current_step exceeds strategy horizon")
  compiled_step = jax.jit(train_step)
  history: list[dict[str, float | int]] = []
  batches = iter(logical_batches)
  for _ in range(start):
    try:
      next(batches)
    except StopIteration as error:
      raise ValueError("logical_batches ends before checkpoint current_step") from error
  for logical_step in range(start, horizon):
    try:
      batch = jax.tree_util.tree_map(jnp.asarray, next(batches))
    except StopIteration as error:
      raise ValueError("logical_batches must contain exactly strategy.horizon batches") from error
    state = compiled_step(state, batch)  # The sole M6 invocation for this B_0.
    current_step = logical_step + 1
    if current_step % eval_every == 0 or current_step == horizon:
      record: dict[str, float | int] = {"step": current_step}
      if evaluate is not None:
        record["accuracy"] = float(evaluate(state))
      history.append(record)
      print(record, flush=True)
      if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            state=state,
            current_step=current_step,
            experiment_config=experiment_config,
            artifact_identifiers=artifact_identifiers,
        )
  try:
    next(batches)
  except StopIteration:
    return state, history
  raise ValueError("logical_batches must contain exactly strategy.horizon batches")


def train_cifar10(config: Cifar10TrainConfig, *, resume_checkpoint: str | Path | None = None):
  """Loads public assets and delegates all private update math to M6."""
  strategy = load_bandinv_strategy(config.strategy)
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_logical_schedule(
      num_examples=len(train_images), batch_size=config.batch_size, strategy=strategy, seed=config.seed
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  params = load_pretrained_vit_tiny(config.pretrained, key=parameter_key)
  participation = ParticipationSpec(strategy.horizon, strategy.min_sep, strategy.max_participations)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,  # type: ignore[arg-type]
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  train_step = make_nonamplified_bandinv_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      strategy,
      calibration,
      participation,
      config.momentum,
      config.learning_rate,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_nonamplified_bandinv_state(params, strategy, noise_key)
  checkpoint_path = Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state,
      train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=strategy.horizon,
      experiment_config=asdict(config),
      artifact_identifiers={"strategy": str(Path(config.strategy).resolve()), "pretrained": str(config.pretrained)},
      checkpoint_path=checkpoint_path,
      resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier(state.params, model, test_images, test_labels, batch_size=config.batch_size),
  )


__all__ = [
    "Cifar10TrainConfig",
    "build_logical_schedule",
    "cross_entropy_loss",
    "evaluate_classifier",
    "run_training",
    "train_cifar10",
]

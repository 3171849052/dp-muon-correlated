"""Shared CIFAR-10 orchestration for non-amplified private trainers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from dp_muon.bandinvmf import BandInvMFStrategy, fit_bandinv_strategy
from dp_muon.data import (
    iter_logical_batches,
    load_cifar10,
    load_public_private_cifar,
    prepare_cifar10_batch,
)
from dp_muon.models import ViTTiny
from dp_muon.optim import PublicVAdamW
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
    calibrate_nonamplified_iid,
    certify_participation_schedule,
    epsilon_spent_for_bandinv_prefix,
    epsilon_spent_for_iid_prefix,
    continuous_hybrid_sensitivity_squared,
    epsilon_spent_for_continuous_hybrid_prefix,
)

from .checkpoint import load_checkpoint, save_checkpoint, validate_resume_identity
from .bandinvmf_strategy_manager import LoadedStrategySnapshot, load_strategy_snapshot
from .pretrained_snapshot import load_pretrained_snapshot
from .run_logging import MetricsCSVWriter
from .nonamplified_linear import (
    NonAmplifiedBandInvState,
    init_nonamplified_bandinv_state,
    make_nonamplified_bandinv_train_step,
)
from .nonamplified_dpsgd import (
    NonAmplifiedDPSGDState,
    init_nonamplified_dpsgd_state,
    make_nonamplified_dpsgd_train_step,
)
from .nonamplified_dpmuon import (
    init_nonamplified_dpmuon_state,
    make_nonamplified_dpmuon_train_step,
)
from .nonamplified_dpadamw import (
    init_nonamplified_dpadamw_state,
    make_nonamplified_dpadamw_train_step,
)
from .nonamplified_bandinv_dpmuon import (
    init_nonamplified_bandinv_dpmuon_state,
    make_nonamplified_bandinv_dpmuon_train_step,
)
from .nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from .nonamplified_segmented_bandinv_dpadamw import (
    SegmentedPlan,
    epsilon_spent_for_segmented_prefix,
    init_segmented_bandinv_dpadamw_state,
    make_segmented_bandinv_dpadamw_train_step,
)
from .nonamplified_frozen_p_bandinv_dpadamw import (
    init_nonamplified_frozen_p_bandinv_dpadamw_state,
    make_nonamplified_frozen_p_bandinv_dpadamw_train_step,
)
from .nonamplified_public_v_bandinv import (
    SegmentedBandInvPrivacyAccountant,
    begin_public_v_segment,
    init_public_v_bandinv_adamw_state,
    make_public_v_bandinv_adamw_train_step,
)
from .public_v import PublicVEstimator


BANDINV_DPMUON_ALGORITHM = "dp-muon-correlated-naive"
BANDINV_DPADAMW_ALGORITHM = "dp-adamw-correlated-naive"
SEGMENTED_BANDINV_DPADAMW_ALGORITHM = "dp-adamw-correlated-segmented"
FROZEN_P_BANDINV_DPADAMW_ALGORITHM = "dp-adamw-correlated-frozen-p"
PUBLIC_V_BANDINV_DPADAMW_ALGORITHM = "dp-adamw-public-v-bandinv"


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


@dataclass(frozen=True)
class Cifar10DPSGDMomentumTrainConfig:
  """Public configuration for the IID DP-SGD-Momentum CIFAR-10 baseline."""

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
  horizon: int
  min_sep: int
  max_participations: int
  adjacency: str = "add_remove"

  def __post_init__(self) -> None:
    if self.batch_size < 1 or self.horizon < 1 or self.min_sep < 1:
      raise ValueError("batch_size, horizon, and min_sep must be positive")
    if self.max_participations < 1:
      raise ValueError("max_participations must be positive")
    if self.microbatch_size is not None and self.microbatch_size < 1:
      raise ValueError("microbatch_size must be positive when supplied")
    if self.microbatch_size is not None and self.batch_size % self.microbatch_size != 0:
      raise ValueError("batch_size must be divisible by microbatch_size")
    if self.eval_every < 1:
      raise ValueError("eval_every must be positive")


@dataclass(frozen=True)
class Cifar10DPMuonTrainConfig:
  """CIFAR-10 configuration for the non-amplified IID DP-Muon baseline."""

  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
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
  seed: int
  checkpoint_dir: str
  eval_every: int
  horizon: int
  min_sep: int
  max_participations: int
  adjacency: str = "add_remove"
  use_bf16_ns: bool = True

  def __post_init__(self) -> None:
    if self.batch_size < 1 or self.horizon < 1 or self.min_sep < 1:
      raise ValueError("batch_size, horizon, and min_sep must be positive")
    if self.max_participations < 1:
      raise ValueError("max_participations must be positive")
    if self.microbatch_size is not None and (
        self.microbatch_size < 1 or self.batch_size % self.microbatch_size
    ):
      raise ValueError("microbatch_size must be positive and divide batch_size")
    if self.eval_every < 1:
      raise ValueError("eval_every must be positive")


@dataclass(frozen=True)
class Cifar10BandInvDPMuonTrainConfig:
  """CIFAR-10 config for one BandInvMF-private Muon/AdamW update stream."""

  strategy: str
  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
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
  seed: int
  checkpoint_dir: str
  eval_every: int
  adjacency: str = "add_remove"
  use_bf16_ns: bool = True

  def __post_init__(self) -> None:
    if self.batch_size < 1:
      raise ValueError("batch_size must be positive")
    if self.microbatch_size is not None and (
        self.microbatch_size < 1 or self.batch_size % self.microbatch_size
    ):
      raise ValueError("microbatch_size must be positive and divide batch_size")
    if self.eval_every < 1:
      raise ValueError("eval_every must be positive")


@dataclass(frozen=True)
class Cifar10BandInvDPAdamWTrainConfig:
  """CIFAR-10 config for one BandInvMF-private AdamW update stream."""

  strategy: str
  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
  learning_rate: float
  beta1: float
  beta2: float
  eps: float
  weight_decay: float
  seed: int
  checkpoint_dir: str
  eval_every: int
  adjacency: str = "add_remove"

  def __post_init__(self) -> None:
    if self.batch_size < 1:
      raise ValueError("batch_size must be positive")
    if self.microbatch_size is not None and (
        self.microbatch_size < 1 or self.batch_size % self.microbatch_size
    ):
      raise ValueError("microbatch_size must be positive and divide batch_size")
    if self.eval_every < 1:
      raise ValueError("eval_every must be positive")


@dataclass(frozen=True)
class Cifar10SegmentedBandInvDPAdamWTrainConfig:
  """CIFAR-10 configuration for segmented correlated DP-AdamW."""

  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
  learning_rate: float
  beta1: float
  beta2: float
  eps: float
  weight_decay: float
  segment_length: int
  bandwidth: int
  reduction: str
  max_optimizer_steps: int
  seed: int
  checkpoint_dir: str
  eval_every: int
  adjacency: str = "add_remove"

  def __post_init__(self) -> None:
    if self.batch_size < 1 or self.segment_length < 1 or self.bandwidth < 1:
      raise ValueError("batch_size, segment_length, and bandwidth must be positive")
    if self.microbatch_size is not None and (
        self.microbatch_size < 1 or self.batch_size % self.microbatch_size
    ):
      raise ValueError("microbatch_size must be positive and divide batch_size")
    if self.max_optimizer_steps < 1 or self.eval_every < 1:
      raise ValueError("max_optimizer_steps and eval_every must be positive")


@dataclass(frozen=True)
class Cifar10FrozenPBandInvDPAdamWTrainConfig:
  """CIFAR-10 config for IID warmup plus continuous frozen-p BandInvMF."""

  strategy: str
  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
  learning_rate: float
  beta1: float
  beta2: float
  eps: float
  weight_decay: float
  seed: int
  checkpoint_dir: str
  eval_every: int
  horizon: int
  min_sep: int
  max_participations: int
  switch_step: int
  adjacency: str = "add_remove"
  warmup_learning_rate: float | None = None

  def __post_init__(self) -> None:
    if self.batch_size < 1 or self.horizon < 1 or self.min_sep < 1:
      raise ValueError("batch_size, horizon, and min_sep must be positive")
    if not 1 <= self.switch_step < self.horizon:
      raise ValueError("switch_step must lie in [1, horizon)")
    if self.max_participations < 1:
      raise ValueError("max_participations must be positive")
    if self.microbatch_size is not None and (
        self.microbatch_size < 1 or self.batch_size % self.microbatch_size
    ):
      raise ValueError("microbatch_size must be positive and divide batch_size")
    if self.eval_every < 1:
      raise ValueError("eval_every must be positive")


@dataclass(frozen=True)
class Cifar10DPAdamWTrainConfig:
  """CIFAR-10 configuration for the non-amplified IID DP-AdamW baseline."""

  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
  learning_rate: float
  beta1: float
  beta2: float
  eps: float
  weight_decay: float
  seed: int
  checkpoint_dir: str
  eval_every: int
  horizon: int
  min_sep: int
  max_participations: int
  adjacency: str = "add_remove"

  def __post_init__(self) -> None:
    if self.batch_size < 1 or self.horizon < 1 or self.min_sep < 1:
      raise ValueError("batch_size, horizon, and min_sep must be positive")
    if self.max_participations < 1:
      raise ValueError("max_participations must be positive")
    if self.microbatch_size is not None and (
        self.microbatch_size < 1 or self.batch_size % self.microbatch_size
    ):
      raise ValueError("microbatch_size must be positive and divide batch_size")
    if self.eval_every < 1:
      raise ValueError("eval_every must be positive")


@dataclass(frozen=True)
class Cifar10PublicVBandInvDPAdamWTrainConfig:
  """CIFAR-10 configuration for segmented Public-(V) BandInvMF AdamW."""

  pretrained: str
  data_dir: str
  batch_size: int
  microbatch_size: int | None
  clip_norm: float
  epsilon: float
  delta: float
  learning_rate: float
  beta1: float
  weight_decay: float
  public_source: str
  cifar10_public_size: int
  public_split_seed: int
  cifar100_public_classes: tuple[int, ...]
  public_v_temporal_mode: str
  public_v_beta2: float
  public_v_eps: float
  public_v_batches_per_segment: int
  segment_length: int
  bandwidth: int
  reduction: str
  max_optimizer_steps: int
  seed: int
  checkpoint_dir: str
  eval_every: int
  horizon: int
  min_sep: int
  max_participations: int
  adjacency: str = "add_remove"

  def __post_init__(self) -> None:
    positive = (
        self.batch_size,
        self.cifar10_public_size,
        self.public_v_batches_per_segment,
        self.segment_length,
        self.bandwidth,
        self.max_optimizer_steps,
        self.eval_every,
        self.horizon,
        self.min_sep,
        self.max_participations,
    )
    if any(value < 1 for value in positive):
      raise ValueError("batch, public-V, strategy, and horizon sizes must be positive")
    if self.microbatch_size is not None and (
        self.microbatch_size < 1 or self.batch_size % self.microbatch_size
    ):
      raise ValueError("microbatch_size must be positive and divide batch_size")
    if self.public_source not in {"cifar10_split", "cifar100_10class"}:
      raise ValueError("public_source is invalid")
    if (
        len(self.cifar100_public_classes) != 10
        or len(set(self.cifar100_public_classes)) != 10
        or any(value < 0 or value >= 100 for value in self.cifar100_public_classes)
    ):
      raise ValueError("cifar100_public_classes must contain 10 unique IDs in [0, 99]")
    if not 0 <= self.beta1 < 1:
      raise ValueError("Adam beta1 must be in [0, 1)")
    if self.public_v_temporal_mode not in {"direct", "ema"}:
      raise ValueError("public_v_temporal_mode must be 'direct' or 'ema'")
    if not 0 <= self.public_v_beta2 < 1:
      raise ValueError("public_v_beta2 must be in [0, 1)")
    if self.learning_rate <= 0 or self.public_v_eps <= 0 or self.weight_decay < 0:
      raise ValueError("AdamW scalar configuration is invalid")


def build_logical_schedule(
    *, num_examples: int, batch_size: int, strategy: BandInvMFStrategy, seed: int
) -> list[np.ndarray]:
  """Builds a deterministic fixed-size schedule and certifies it against M4."""
  if num_examples < 1 or batch_size < 1:
    raise ValueError("num_examples and batch_size must be positive")
  if batch_size > num_examples:
    raise ValueError("batch_size must not exceed number of training examples")
  return build_fixed_cycle_logical_schedule(
      num_examples=num_examples,
      batch_size=batch_size,
      horizon=strategy.horizon,
      min_sep=strategy.min_sep,
      max_participations=strategy.max_participations,
      seed=seed,
  )


def build_fixed_cycle_logical_schedule(
    *,
    num_examples: int,
    batch_size: int,
    horizon: int,
    min_sep: int,
    max_participations: int | None,
    seed: int,
) -> list[np.ndarray]:
  """Builds and certifies the shared fixed-cycle schedule without a strategy."""
  if num_examples < 1 or batch_size < 1 or horizon < 1 or min_sep < 1:
    raise ValueError("schedule dimensions must be positive")
  if batch_size > num_examples:
    raise ValueError("batch_size must not exceed number of training examples")
  # Moving through one shuffled cyclic order makes every repeated record at
  # least floor(num_examples / batch_size) steps apart.  Certification below is
  # the authority for the fitted min-separation/max-participation contract.
  permutation = np.random.default_rng(seed).permutation(num_examples)
  schedule = [
      permutation[(step * batch_size + np.arange(batch_size)) % num_examples].astype(np.int32)
      for step in range(horizon)
  ]
  certify_participation_schedule(
      schedule, ParticipationSpec(horizon, min_sep, max_participations)
  )
  return schedule


def cross_entropy_loss(params: dict, batch: dict[str, jax.Array], model: ViTTiny) -> jax.Array:
  logits = model.apply(params, batch["image"])
  labels = batch["label"]
  return -jax.nn.log_softmax(logits)[0, labels[0]]


def public_cross_entropy_loss(
    params: dict, batch: dict[str, jax.Array], model: ViTTiny
) -> jax.Array:
  """Returns the mean public-batch loss used only for V estimation."""
  logits = model.apply(params, batch["image"])
  log_probabilities = jax.nn.log_softmax(logits, axis=-1)
  return -jnp.mean(
      jnp.take_along_axis(log_probabilities, batch["label"][:, None], axis=-1)
  )


def evaluate_classifier_metrics(
    params: dict, model: ViTTiny, images: np.ndarray, labels: np.ndarray, *, batch_size: int
) -> dict[str, float]:
  """Computes test loss and accuracy together from one forward per batch."""
  correct = 0
  total_loss = 0.0
  for offset in tqdm(
      range(0, len(images), batch_size), desc="Evaluating", unit="batch"
  ):
    batch = prepare_cifar10_batch(images[offset : offset + batch_size], labels[offset : offset + batch_size])
    logits = model.apply(params, jnp.asarray(batch["image"]))
    batch_labels = jnp.asarray(batch["label"])
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    total_loss -= float(jnp.sum(jnp.take_along_axis(
        log_probabilities, batch_labels[:, None], axis=-1
    )))
    predictions = np.asarray(jnp.argmax(logits, axis=-1))
    correct += int(np.sum(predictions == batch["label"]))
  return {"test_loss": total_loss / len(images), "test_accuracy": correct / len(images)}


def evaluate_classifier(
    params: dict, model: ViTTiny, images: np.ndarray, labels: np.ndarray, *, batch_size: int
) -> float:
  """Compatibility wrapper for callers that only need test accuracy."""
  return evaluate_classifier_metrics(
      params, model, images, labels, batch_size=batch_size
  )["test_accuracy"]


def run_training(
    *,
    initial_state: Any,
    train_step: Callable[[Any, Any], Any],
    logical_batches: Iterable[Any],
    horizon: int,
    experiment_config: dict[str, Any],
    artifact_identifiers: dict[str, str],
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    eval_every: int = 1,
    evaluate: Callable[[Any], Mapping[str, float] | float] | None = None,
    num_train_examples: int | None = None,
    logical_batch_size: int | None = None,
    metrics_writer: MetricsCSVWriter | None = None,
    privacy_accountant: Callable[[int], float] | None = None,
    before_step: Callable[[Any, int], Any] | None = None,
    after_step: Callable[[Any, int], None] | None = None,
    on_state_ready: Callable[[Any, int], None] | None = None,
) -> tuple[Any, list[dict[str, float | int]]]:
  """Runs exactly one private update for each logical batch, with optional resume.

  The iterable is deliberately consumed one batch at a time.  No accumulation,
  no splitting, and no retry/skip path can introduce another DP query.
  """
  if horizon < 1 or eval_every < 1:
    raise ValueError("horizon and eval_every must be positive")
  if (num_train_examples is None) != (logical_batch_size is None):
    raise ValueError("num_train_examples and logical_batch_size must be supplied together")
  if num_train_examples is not None and (
      num_train_examples < 1 or logical_batch_size is None or logical_batch_size < 1
  ):
    raise ValueError("epoch progress dimensions must be positive")
  state, start = initial_state, 0
  if resume_checkpoint is not None:
    saved = load_checkpoint(resume_checkpoint)
    validate_resume_identity(
        saved, experiment_config=experiment_config,
        artifact_identifiers=artifact_identifiers,
    )
    state, start = saved["state"], int(saved["current_step"])
    if start > horizon:
      raise ValueError("checkpoint current_step exceeds training horizon")
  if on_state_ready is not None:
    on_state_ready(state, start)
  compiled_step = jax.jit(train_step)
  history: list[dict[str, float | int]] = []
  started_at = time.monotonic()
  next_eval_epoch = 0
  last_recorded_epoch = 0
  if num_train_examples is not None and logical_batch_size is not None:
    completed_at_start = math.floor(start * logical_batch_size / num_train_examples)
    next_eval_epoch = ((completed_at_start // eval_every) + 1) * eval_every
    last_recorded_epoch = completed_at_start
  batches = iter(logical_batches)
  for _ in range(start):
    try:
      next(batches)
    except StopIteration as error:
      raise ValueError("logical_batches ends before checkpoint current_step") from error
  with tqdm(
      range(start, horizon),
      total=horizon,
      initial=start,
      desc="Training",
      unit="logical batch",
  ) as progress:
    for logical_step in progress:
      if before_step is not None:
        state = before_step(state, logical_step)
      try:
        batch = jax.tree_util.tree_map(jnp.asarray, next(batches))
      except StopIteration as error:
        raise ValueError("logical_batches must contain exactly horizon batches") from error
      state = compiled_step(state, batch)  # The sole private update for this batch.
      current_step = logical_step + 1
      if after_step is not None:
        after_step(state, current_step)
      if num_train_examples is None:
        should_evaluate = current_step % eval_every == 0 or current_step == horizon
      else:
        assert logical_batch_size is not None
        effective_epoch = current_step * logical_batch_size / num_train_examples
        should_evaluate = (
            effective_epoch + 1e-12 >= next_eval_epoch
            or current_step == horizon
        )
      if should_evaluate:
        evaluation_epoch: int | None = None
        if num_train_examples is not None:
          # ``next_eval_epoch`` is the integer epoch just crossed.  In
          # particular, a step at effective epoch 1.00352 records epoch 1,
          # rather than using ceil() and incorrectly recording epoch 2.
          if effective_epoch + 1e-12 >= next_eval_epoch:
            evaluation_epoch = next_eval_epoch
          else:
            # The final horizon can be fractional.  Give this forced eval the
            # next meaningful label without colliding with a prior boundary.
            evaluation_epoch = max(
                last_recorded_epoch + 1,
                math.ceil(effective_epoch - 1e-12),
            )
        record: dict[str, float | int] = {"step": current_step}
        if evaluate is not None:
          evaluation_started_at = time.monotonic()
          result = evaluate(state)
          eval_seconds = time.monotonic() - evaluation_started_at
          if isinstance(result, Mapping):
            record.update({key: float(value) for key, value in result.items()})
          else:
            record["accuracy"] = float(result)
        else:
          eval_seconds = 0.0
        if num_train_examples is not None:
          assert logical_batch_size is not None
          effective_epoch = current_step * logical_batch_size / num_train_examples
          assert evaluation_epoch is not None
          epoch = evaluation_epoch
          metrics_record: dict[str, float | int] = {
              "epoch": epoch,
              "step": current_step,
              "effective_epoch": effective_epoch,
              "epsilon_spent": (
                  float(privacy_accountant(current_step))
                  if privacy_accountant is not None else float("nan")
              ),
              "test_loss": float(record.get("test_loss", float("nan"))),
              "test_accuracy": float(record.get("test_accuracy", record.get("accuracy", float("nan")))),
              "elapsed_seconds": time.monotonic() - started_at,
              "eval_seconds": eval_seconds,
          }
          if metrics_writer is not None:
            metrics_writer.append(metrics_record)
          record.update(metrics_record)
          last_recorded_epoch = epoch
          while effective_epoch + 1e-12 >= next_eval_epoch:
            next_eval_epoch += eval_every
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
  raise ValueError("logical_batches must contain exactly horizon batches")


def train_cifar10(
    config: Cifar10TrainConfig,
    *,
    strategy_snapshot: LoadedStrategySnapshot | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Loads public assets and delegates all private update math to M6."""
  strategy_snapshot = strategy_snapshot or load_strategy_snapshot(config.strategy)
  strategy = strategy_snapshot.strategy
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_logical_schedule(
      num_examples=len(train_images), batch_size=config.batch_size, strategy=strategy, seed=config.seed
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  params = pretrained_snapshot.params
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
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state,
      train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=strategy.horizon,
      experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": "bandinv",
          "strategy_path": str(strategy_snapshot.path),
          "strategy_sha256": strategy_snapshot.sha256,
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path,
      resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(
          state.params, model, test_images, test_labels, batch_size=config.batch_size
      ),
      num_train_examples=len(train_images),
      logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=lambda step: epsilon_spent_for_bandinv_prefix(
          prefix_steps=step,
          noising_coef=strategy.noising_coef,
          horizon=strategy.horizon,
          min_sep=strategy.min_sep,
          max_participations=strategy.max_participations,
          calibration=calibration,
          full_sensitivity_squared=float(strategy.sensitivity_squared),
      ),
  )


def train_cifar10_dpsgd_momentum(
    config: Cifar10DPSGDMomentumTrainConfig,
    *,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Fine-tunes CIFAR-10 with the non-amplified IID DP-SGD baseline."""
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images),
      batch_size=config.batch_size,
      horizon=config.horizon,
      min_sep=config.min_sep,
      max_participations=config.max_participations,
      seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  params = pretrained_snapshot.params
  calibration = calibrate_nonamplified_iid(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,  # type: ignore[arg-type]
      max_participations=config.max_participations,
  )
  train_step = make_nonamplified_dpsgd_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      calibration,
      config.momentum,
      config.learning_rate,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_nonamplified_dpsgd_state(params, noise_key)
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state,
      train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=config.horizon,
      experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": "nonamplified_iid_dpsgd_momentum",
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path,
      resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(
          state.params, model, test_images, test_labels, batch_size=config.batch_size
      ),
      num_train_examples=len(train_images),
      logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=lambda step: epsilon_spent_for_iid_prefix(
          prefix_steps=step,
          horizon=config.horizon,
          min_sep=config.min_sep,
          max_participations=config.max_participations,
          calibration=calibration,
      ),
  )


def train_cifar10_dpmuon(
    config: Cifar10DPMuonTrainConfig,
    *,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Fine-tunes CIFAR-10 using one IID-private gradient per logical batch."""
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images), batch_size=config.batch_size,
      horizon=config.horizon, min_sep=config.min_sep,
      max_participations=config.max_participations, seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  params = pretrained_snapshot.params
  calibration = calibrate_nonamplified_iid(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,  # type: ignore[arg-type]
      max_participations=config.max_participations,
  )
  train_step, optimizer = make_nonamplified_dpmuon_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model), calibration,
      muon_learning_rate=config.muon_learning_rate,
      muon_weight_decay=config.muon_weight_decay, momentum=config.momentum,
      ns_steps=config.ns_steps, consistent_rms=config.consistent_rms,
      adamw_learning_rate=config.adamw_learning_rate,
      adamw_beta1=config.adamw_beta1, adamw_beta2=config.adamw_beta2,
      adamw_eps=config.adamw_eps, adamw_weight_decay=config.adamw_weight_decay,
      microbatch_size=config.microbatch_size, use_bf16_ns=config.use_bf16_ns,
  )
  initial_state = init_nonamplified_dpmuon_state(params, noise_key, optimizer)
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state, train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=config.horizon, experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": "nonamplified_iid_dpmuon",
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path, resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(state.params, model, test_images, test_labels, batch_size=config.batch_size),
      num_train_examples=len(train_images), logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=lambda step: epsilon_spent_for_iid_prefix(
          prefix_steps=step, horizon=config.horizon, min_sep=config.min_sep,
          max_participations=config.max_participations, calibration=calibration,
      ),
  )


def train_cifar10_dpadamw(
    config: Cifar10DPAdamWTrainConfig,
    *,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Fine-tunes CIFAR-10 using one IID-private gradient, all-parameter AdamW."""
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images), batch_size=config.batch_size,
      horizon=config.horizon, min_sep=config.min_sep,
      max_participations=config.max_participations, seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  params = pretrained_snapshot.params
  calibration = calibrate_nonamplified_iid(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,  # type: ignore[arg-type]
      max_participations=config.max_participations,
  )
  train_step, optimizer = make_nonamplified_dpadamw_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model), calibration,
      learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2,
      eps=config.eps, weight_decay=config.weight_decay,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_nonamplified_dpadamw_state(params, noise_key, optimizer)
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state, train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=config.horizon, experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": "nonamplified_iid_dpadamw",
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path, resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(state.params, model, test_images, test_labels, batch_size=config.batch_size),
      num_train_examples=len(train_images), logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=lambda step: epsilon_spent_for_iid_prefix(
          prefix_steps=step, horizon=config.horizon, min_sep=config.min_sep,
          max_participations=config.max_participations, calibration=calibration,
      ),
  )


def train_cifar10_bandinv_dpmuon(
    config: Cifar10BandInvDPMuonTrainConfig,
    *,
    strategy_snapshot: LoadedStrategySnapshot | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Fine-tunes CIFAR-10 with one full-tree BandInvMF private gradient."""
  strategy_snapshot = strategy_snapshot or load_strategy_snapshot(config.strategy)
  strategy = strategy_snapshot.strategy
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_logical_schedule(
      num_examples=len(train_images), batch_size=config.batch_size,
      strategy=strategy, seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  params = pretrained_snapshot.params
  participation = ParticipationSpec(
      strategy.horizon, strategy.min_sep, strategy.max_participations
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,  # type: ignore[arg-type]
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  train_step, optimizer = make_nonamplified_bandinv_dpmuon_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      strategy, calibration, participation,
      muon_learning_rate=config.muon_learning_rate,
      muon_weight_decay=config.muon_weight_decay,
      momentum=config.momentum, ns_steps=config.ns_steps,
      consistent_rms=config.consistent_rms,
      adamw_learning_rate=config.adamw_learning_rate,
      adamw_beta1=config.adamw_beta1, adamw_beta2=config.adamw_beta2,
      adamw_eps=config.adamw_eps, adamw_weight_decay=config.adamw_weight_decay,
      microbatch_size=config.microbatch_size, use_bf16_ns=config.use_bf16_ns,
  )
  initial_state = init_nonamplified_bandinv_dpmuon_state(
      params, strategy, noise_key, optimizer
  )
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state, train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=strategy.horizon, experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": BANDINV_DPMUON_ALGORITHM,
          "strategy_path": str(strategy_snapshot.path),
          "strategy_sha256": strategy_snapshot.sha256,
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path, resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(
          state.params, model, test_images, test_labels, batch_size=config.batch_size
      ),
      num_train_examples=len(train_images), logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=lambda step: epsilon_spent_for_bandinv_prefix(
          prefix_steps=step, noising_coef=strategy.noising_coef,
          horizon=strategy.horizon, min_sep=strategy.min_sep,
          max_participations=strategy.max_participations, calibration=calibration,
          full_sensitivity_squared=float(strategy.sensitivity_squared),
      ),
  )


def train_cifar10_bandinv_dpadamw(
    config: Cifar10BandInvDPAdamWTrainConfig,
    *,
    strategy_snapshot: LoadedStrategySnapshot | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Fine-tunes CIFAR-10 with one full-tree BandInvMF private gradient,
  all-parameter AdamW."""
  strategy_snapshot = strategy_snapshot or load_strategy_snapshot(config.strategy)
  strategy = strategy_snapshot.strategy
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_logical_schedule(
      num_examples=len(train_images), batch_size=config.batch_size,
      strategy=strategy, seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(config.pretrained, key=parameter_key)
  params = pretrained_snapshot.params
  participation = ParticipationSpec(
      strategy.horizon, strategy.min_sep, strategy.max_participations
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,  # type: ignore[arg-type]
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  train_step, optimizer = make_nonamplified_bandinv_dpadamw_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      strategy, calibration, participation,
      learning_rate=config.learning_rate,
      beta1=config.beta1, beta2=config.beta2,
      eps=config.eps, weight_decay=config.weight_decay,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_nonamplified_bandinv_dpadamw_state(
      params, strategy, noise_key, optimizer
  )
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state, train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=strategy.horizon, experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": BANDINV_DPADAMW_ALGORITHM,
          "strategy_path": str(strategy_snapshot.path),
          "strategy_sha256": strategy_snapshot.sha256,
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path, resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(
          state.params, model, test_images, test_labels, batch_size=config.batch_size
      ),
      num_train_examples=len(train_images), logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=lambda step: epsilon_spent_for_bandinv_prefix(
          prefix_steps=step, noising_coef=strategy.noising_coef,
          horizon=strategy.horizon, min_sep=strategy.min_sep,
          max_participations=strategy.max_participations, calibration=calibration,
          full_sensitivity_squared=float(strategy.sensitivity_squared),
      ),
  )


def train_cifar10_segmented_bandinv_dpadamw(
    config: Cifar10SegmentedBandInvDPAdamWTrainConfig,
    plan: SegmentedPlan,
    *,
    strategy_snapshots: Mapping[int, LoadedStrategySnapshot] | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Fine-tunes CIFAR-10 with block-diagonal correlated DP-AdamW noise."""
  if plan.horizon < 1 or plan.global_min_sep is None:
    raise ValueError("segmented plan must contain global horizon and min_sep")
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images),
      batch_size=config.batch_size,
      horizon=plan.horizon,
      min_sep=plan.global_min_sep,
      max_participations=plan.max_participations,
      seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(path=config.pretrained, key=parameter_key)
  train_step, optimizer = make_segmented_bandinv_dpadamw_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      plan,
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      beta2=config.beta2,
      eps=config.eps,
      weight_decay=config.weight_decay,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_segmented_bandinv_dpadamw_state(
      pretrained_snapshot.params, plan, noise_key, optimizer
  )
  artifact_identifiers: dict[str, str] = {
      "algorithm": SEGMENTED_BANDINV_DPADAMW_ALGORITHM,
      "segment_lengths": ",".join(str(length) for length in plan.block_lengths),
      "pretrained_path": str(pretrained_snapshot.path),
      "pretrained_sha256": pretrained_snapshot.sha256,
  }
  if strategy_snapshots:
    for length in sorted(strategy_snapshots):
      snapshot = strategy_snapshots[length]
      artifact_identifiers[f"strategy_{length}_path"] = str(snapshot.path)
      artifact_identifiers[f"strategy_{length}_sha256"] = snapshot.sha256
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state,
      train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=plan.horizon,
      experiment_config=asdict(config),
      artifact_identifiers=artifact_identifiers,
      checkpoint_path=actual_checkpoint_path,
      resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(
          state.params, model, test_images, test_labels, batch_size=config.batch_size
      ),
      num_train_examples=len(train_images),
      logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      # This is an accounting view of the one full-transcript calibration;
      # it never allocates a separate epsilon/mu to an individual segment.
      privacy_accountant=lambda step: epsilon_spent_for_segmented_prefix(plan, step),
  )


def train_cifar10_frozen_p_bandinv_dpadamw(
    config: Cifar10FrozenPBandInvDPAdamWTrainConfig,
    *,
    strategy_snapshot: LoadedStrategySnapshot | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
):
  """Runs IID DP-AdamW warmup, then one continuous frozen-p BandInvMF stream."""
  strategy_snapshot = strategy_snapshot or load_strategy_snapshot(config.strategy)
  strategy = strategy_snapshot.strategy
  if strategy.horizon != config.horizon - config.switch_step:
    raise ValueError("strategy horizon must equal the configured Phase-II horizon")
  train_images, train_labels = load_cifar10(config.data_dir, train=True)
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(train_images),
      batch_size=config.batch_size,
      horizon=config.horizon,
      min_sep=config.min_sep,
      max_participations=config.max_participations,
      seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(path=config.pretrained, key=parameter_key)
  participation = ParticipationSpec(
      config.horizon, config.min_sep, config.max_participations
  )
  hybrid_sensitivity_squared = continuous_hybrid_sensitivity_squared(
      config.switch_step,
      strategy,
      min_sep=config.min_sep,
      max_participations=config.max_participations,
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,  # type: ignore[arg-type]
      sensitivity_squared=hybrid_sensitivity_squared,
  )
  train_step, optimizer = make_nonamplified_frozen_p_bandinv_dpadamw_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      strategy,
      calibration,
      participation,
      switch_step=config.switch_step,
      learning_rate=config.learning_rate,
      warmup_learning_rate=config.warmup_learning_rate,
      beta1=config.beta1,
      beta2=config.beta2,
      eps=config.eps,
      weight_decay=config.weight_decay,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_nonamplified_frozen_p_bandinv_dpadamw_state(
      pretrained_snapshot.params,
      strategy,
      noise_key,
      optimizer,
      switch_step=config.switch_step,
  )
  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state,
      train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=config.horizon,
      experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": FROZEN_P_BANDINV_DPADAMW_ALGORITHM,
          "strategy_path": str(strategy_snapshot.path),
          "strategy_sha256": strategy_snapshot.sha256,
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path,
      resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(
          state.params, model, test_images, test_labels, batch_size=config.batch_size
      ),
      num_train_examples=len(train_images),
      logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=lambda step: epsilon_spent_for_continuous_hybrid_prefix(
          prefix_steps=step,
          tau=config.switch_step,
          phase_strategy=strategy,
          min_sep=config.min_sep,
          max_participations=config.max_participations,
          calibration=calibration,
          full_sensitivity_squared=hybrid_sensitivity_squared,
      ),
  )


def _public_batches_for_segment(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    num_batches: int,
    seed: int,
    segment_index: int,
) -> list[dict[str, jax.Array]]:
  """Creates deterministic public batches without private loader state."""
  if len(images) < 1:
    raise ValueError("public dataset must not be empty")
  permutation = np.random.default_rng([seed, segment_index]).permutation(len(images))
  batches = []
  for batch_index in range(num_batches):
    indices = permutation[
        (batch_index * batch_size + np.arange(batch_size)) % len(images)
    ]
    batches.append(
        jax.tree_util.tree_map(
            jnp.asarray,
            prepare_cifar10_batch(images[indices], labels[indices]),
        )
    )
  return batches


def train_dp_public_v_bandinv(
    config: Cifar10PublicVBandInvDPAdamWTrainConfig,
    *,
    resume_checkpoint: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
):
  """Runs Public-(V) + Frozen AdamW + independently segmented BandInvMF."""
  data = load_public_private_cifar(
      config.data_dir,
      public_source=config.public_source,  # type: ignore[arg-type]
      cifar10_public_size=config.cifar10_public_size,
      public_split_seed=config.public_split_seed,
      cifar100_public_classes=config.cifar100_public_classes,
  )
  test_images, test_labels = load_cifar10(config.data_dir, train=False)
  schedule = build_fixed_cycle_logical_schedule(
      num_examples=len(data.private_images),
      batch_size=config.batch_size,
      horizon=config.horizon,
      min_sep=config.min_sep,
      max_participations=config.max_participations,
      seed=config.seed,
  )
  model = ViTTiny()
  parameter_key, noise_root_key = jax.random.split(jax.random.key(config.seed))
  pretrained_snapshot = load_pretrained_snapshot(path=config.pretrained, key=parameter_key)
  optimizer = PublicVAdamW(
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      eps=config.public_v_eps,
      weight_decay=config.weight_decay,
  )
  estimator = PublicVEstimator(
      lambda parameters, batch: public_cross_entropy_loss(parameters, batch, model),
      eps=config.public_v_eps,
  )
  compiled_public_v_batch_estimate = jax.jit(estimator.squared_batch_gradient)

  # The repository's fixed-cycle accountant is non-amplified.  Deriving
  # horizon/min_sep/max_participations from the private subset is what makes
  # both its participation contract and any reported sample rate private-size
  # aware; public examples never enter this calibration.
  clipping_calibration = calibrate_nonamplified_iid(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,  # type: ignore[arg-type]
      max_participations=config.max_participations,
  )
  train_step = make_public_v_bandinv_adamw_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      clipping_calibration,
      optimizer,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_public_v_bandinv_adamw_state(
      pretrained_snapshot.params,
      optimizer=optimizer,
      noise_root_key=noise_root_key,
      bandwidth=min(config.bandwidth, config.segment_length, config.horizon),
  )
  num_segments = math.ceil(config.horizon / config.segment_length)
  accountant = SegmentedBandInvPrivacyAccountant(
      num_segments=num_segments,
      global_mu=clipping_calibration.mu,
      delta=config.delta,
  )

  def start_segment(state: Any, logical_step: int):
    if logical_step % config.segment_length:
      return state
    segment_index = logical_step // config.segment_length
    length = min(config.segment_length, config.horizon - logical_step)
    public_batches = _public_batches_for_segment(
        data.public_images,
        data.public_labels,
        batch_size=config.batch_size,
        num_batches=config.public_v_batches_per_segment,
        seed=config.public_split_seed,
        segment_index=segment_index,
    )
    state, info = begin_public_v_segment(
        state,
        public_batches,
        estimator=estimator,
        optimizer=optimizer,
        segment_index=segment_index,
        segment_length=length,
        global_min_sep=config.min_sep,
        bandwidth=config.bandwidth,
        num_segments=num_segments,
        global_noise_multiplier=clipping_calibration.noise_multiplier,
        query_sensitivity=clipping_calibration.query_sensitivity,
        learning_rates=config.learning_rate,
        reduction=config.reduction,
        max_optimizer_steps=config.max_optimizer_steps,
        temporal_mode=config.public_v_temporal_mode,
        public_v_beta2=config.public_v_beta2,
        fit_strategy=fit_strategy,
        public_v_batch_estimate=compiled_public_v_batch_estimate,
    )
    accountant.set_current_state(state)
    print(
        {
            "segment": info.segment_index,
            "start_step": info.start_step,
            "length": info.length,
            "public_v_num_examples": info.public_v_num_examples,
            "preconditioner_rms": info.preconditioner_rms,
            "strategy_sensitivity_squared": float(info.strategy.sensitivity_squared),
        },
        flush=True,
    )
    return state

  def register_resumed_state(state: Any, start: int) -> None:
    if start > 0 and int(state.segment_index) >= 0:
      accountant.set_current_state(state)

  actual_checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / "latest.pkl"
  return run_training(
      initial_state=initial_state,
      train_step=train_step,
      logical_batches=iter_logical_batches(
          data.private_images, data.private_labels, schedule
      ),
      horizon=config.horizon,
      experiment_config=asdict(config),
      artifact_identifiers={
          "algorithm": PUBLIC_V_BANDINV_DPADAMW_ALGORITHM,
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=actual_checkpoint_path,
      resume_checkpoint=resume_checkpoint,
      eval_every=config.eval_every,
      evaluate=lambda state: evaluate_classifier_metrics(
          state.params,
          model,
          test_images,
          test_labels,
          batch_size=config.batch_size,
      ),
      num_train_examples=len(data.private_images),
      logical_batch_size=config.batch_size,
      metrics_writer=MetricsCSVWriter(metrics_path) if metrics_path is not None else None,
      privacy_accountant=accountant.epsilon_spent,
      before_step=start_segment,
      on_state_ready=register_resumed_state,
  )


__all__ = [
    "BANDINV_DPMUON_ALGORITHM",
    "BANDINV_DPADAMW_ALGORITHM",
    "SEGMENTED_BANDINV_DPADAMW_ALGORITHM",
    "FROZEN_P_BANDINV_DPADAMW_ALGORITHM",
    "PUBLIC_V_BANDINV_DPADAMW_ALGORITHM",
    "Cifar10TrainConfig",
    "Cifar10DPSGDMomentumTrainConfig",
    "Cifar10DPMuonTrainConfig",
    "Cifar10DPAdamWTrainConfig",
    "Cifar10BandInvDPMuonTrainConfig",
    "Cifar10BandInvDPAdamWTrainConfig",
    "Cifar10SegmentedBandInvDPAdamWTrainConfig",
    "Cifar10FrozenPBandInvDPAdamWTrainConfig",
    "Cifar10PublicVBandInvDPAdamWTrainConfig",
    "build_fixed_cycle_logical_schedule",
    "build_logical_schedule",
    "cross_entropy_loss",
    "public_cross_entropy_loss",
    "evaluate_classifier",
    "evaluate_classifier_metrics",
    "run_training",
    "train_cifar10",
    "train_cifar10_dpsgd_momentum",
    "train_cifar10_dpmuon",
    "train_cifar10_dpadamw",
    "train_cifar10_bandinv_dpmuon",
    "train_cifar10_bandinv_dpadamw",
    "train_cifar10_segmented_bandinv_dpadamw",
    "train_cifar10_frozen_p_bandinv_dpadamw",
    "train_dp_public_v_bandinv",
]

"""Participation schedules and certificates for BandInvMF constraints.

The schedule contract is stated for one privacy unit: one training record.
This module uses JAX Privacy's cyclic batch-selection primitive to construct a
baseline schedule and performs eager certification of concrete index batches.
It performs no privacy amplification accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
from jax_privacy import batch_selection
from jax_privacy.matrix_factorization import sensitivity


@dataclass(frozen=True)
class ParticipationSpec:
  """The ``(n, k, b)`` participation contract used by a strategy."""

  horizon: int
  min_sep: int
  max_participations: int | None

  def __post_init__(self) -> None:
    if not isinstance(self.horizon, Integral) or self.horizon < 1:
      raise ValueError("horizon must be a positive integer")
    if not isinstance(self.min_sep, Integral) or self.min_sep < 1:
      raise ValueError("min_sep must be a positive integer")
    if self.max_participations is not None and (
        not isinstance(self.max_participations, Integral) or self.max_participations < 1
    ):
      raise ValueError("max_participations must be a positive integer when supplied")


@dataclass(frozen=True)
class ParticipationCertificate:
  """Aggregate confirmation that a concrete schedule meets a spec."""

  horizon: int
  required_min_sep: int
  observed_min_sep: int | None
  required_max_participations: int | None
  observed_max_participations: int
  valid: bool


def _partition_type_for_adjacency(adjacency: str) -> batch_selection.PartitionType:
  if adjacency == "add_remove":
    return batch_selection.PartitionType.INDEPENDENT
  if adjacency == "replace_one":
    return batch_selection.PartitionType.EQUAL_SPLIT
  raise ValueError("adjacency must be 'add_remove' or 'replace_one'")


def validate_fixed_cycle_dataset(
    *, num_examples: int, min_sep: int, adjacency: str
) -> None:
  """Checks whether a dataset can be used without fixed-cycle record loss.

  JAX Privacy's ``EQUAL_SPLIT`` partition intentionally discards the remainder
  when the dataset size is not divisible by the cycle length.  That behavior is
  incompatible with this baseline's complete-dataset replace-one semantics.
  """
  if not isinstance(num_examples, Integral) or num_examples < 1:
    raise ValueError("num_examples must be a positive integer")
  ParticipationSpec(horizon=1, min_sep=min_sep, max_participations=None)
  _partition_type_for_adjacency(adjacency)
  if adjacency == "replace_one" and num_examples % min_sep != 0:
    raise ValueError(
        "replace_one fixed-cycle baseline requires num_examples % min_sep == 0; "
        "JAX Privacy EQUAL_SPLIT would otherwise leave remainder records never participating"
    )


def make_fixed_cycle_selection(
    *, horizon: int, min_sep: int, adjacency: str, num_examples: int | None = None
) -> batch_selection.CyclicPoissonSampling:
  """Returns JAX Privacy's full-probability fixed-cycle selector.

  Every record belongs to one of ``min_sep`` fixed cycles; hence its possible
  appearances are separated by at least ``min_sep`` steps.  This is only a
  schedule primitive, not an amplification mechanism.
  """
  ParticipationSpec(horizon=horizon, min_sep=min_sep, max_participations=None)
  partition_type = _partition_type_for_adjacency(adjacency)
  if num_examples is not None:
    validate_fixed_cycle_dataset(
        num_examples=num_examples, min_sep=min_sep, adjacency=adjacency
    )
  return batch_selection.CyclicPoissonSampling(
      sampling_prob=1.0,
      iterations=horizon,
      cycle_length=min_sep,
      partition_type=partition_type,
  )


def theoretical_max_participations(spec: ParticipationSpec) -> int:
  """Returns JAX Privacy's `(n, b, k)` participation upper bound."""
  return sensitivity.minsep_true_max_participations(
      n=spec.horizon,
      min_sep=spec.min_sep,
      max_participations=spec.max_participations,
  )


def certify_participation_schedule(
    batches: Iterable[object], spec: ParticipationSpec
) -> ParticipationCertificate:
  """Fail-fast certifies concrete batches against ``spec``.

  ``-1`` is a padding sentinel and is ignored.  The certificate keeps only
  aggregate extrema; temporary counts and last-seen steps are discarded before
  returning, so no per-record participation history is retained.
  """
  counts: dict[int, int] = {}
  last_seen: dict[int, int] = {}
  observed_min_sep: int | None = None
  observed_max_participations = 0
  observed_horizon = 0
  for step, batch in enumerate(batches):
    if step >= spec.horizon:
      raise ValueError("number of batches must equal spec.horizon")
    observed_horizon = step + 1
    indices = np.asarray(batch)
    if indices.ndim != 1:
      raise ValueError("each batch must be a one-dimensional index array")
    if not np.issubdtype(indices.dtype, np.integer):
      raise ValueError("batch indices must be integers")
    records = [int(index) for index in indices if index != -1]
    if any(index < 0 for index in records):
      raise ValueError("only -1 may be used as a negative padding index")
    if len(records) != len(set(records)):
      raise ValueError("a record may appear at most once within a batch")
    for record in records:
      if record in last_seen:
        gap = step - last_seen[record]
        if gap < spec.min_sep:
          raise ValueError("consecutive participations violate min_sep")
        observed_min_sep = gap if observed_min_sep is None else min(observed_min_sep, gap)
      last_seen[record] = step
      counts[record] = counts.get(record, 0) + 1
      observed_max_participations = max(observed_max_participations, counts[record])
      if spec.max_participations is not None and counts[record] > spec.max_participations:
        raise ValueError("participations exceed max_participations")

  if observed_horizon != spec.horizon:
    raise ValueError("number of batches must equal spec.horizon")

  return ParticipationCertificate(
      horizon=spec.horizon,
      required_min_sep=spec.min_sep,
      observed_min_sep=observed_min_sep,
      required_max_participations=spec.max_participations,
      observed_max_participations=observed_max_participations,
      valid=True,
  )


def participation_spec_from_strategy(strategy: object) -> ParticipationSpec:
  """Extracts the exact participation contract embedded in a BandInvMF strategy."""
  try:
    return ParticipationSpec(
        horizon=strategy.horizon,  # type: ignore[attr-defined]
        min_sep=strategy.min_sep,  # type: ignore[attr-defined]
        max_participations=strategy.max_participations,  # type: ignore[attr-defined]
    )
  except AttributeError as error:
    raise TypeError("strategy must expose horizon, min_sep, and max_participations") from error


def validate_participation_spec_against_strategy(spec: ParticipationSpec, strategy: object) -> None:
  """Raises unless schedule and strategy use identical `(n, k, b)` values."""
  strategy_spec = participation_spec_from_strategy(strategy)
  if spec != strategy_spec:
    raise ValueError("participation spec must exactly match strategy horizon, min_sep, and max_participations")


__all__ = [
    "ParticipationCertificate",
    "ParticipationSpec",
    "certify_participation_schedule",
    "make_fixed_cycle_selection",
    "participation_spec_from_strategy",
    "theoretical_max_participations",
    "validate_fixed_cycle_dataset",
    "validate_participation_spec_against_strategy",
]

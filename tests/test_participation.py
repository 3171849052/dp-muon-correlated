"""Tests for BandInvMF participation schedules and certificates."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy import batch_selection

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import (
    ParticipationSpec,
    certify_participation_schedule,
    make_fixed_cycle_selection,
    participation_spec_from_strategy,
    theoretical_max_participations,
    validate_participation_spec_against_strategy,
)


def test_fixed_cycle_schedule_is_certified_and_has_minimum_separation():
  spec = ParticipationSpec(horizon=10, min_sep=3, max_participations=None)
  selector = make_fixed_cycle_selection(horizon=spec.horizon, min_sep=spec.min_sep, adjacency="add_remove")
  batches = list(selector.batch_iterator(num_examples=11, rng=123))
  certificate = certify_participation_schedule(batches, spec)
  assert len(batches) == 10
  assert certificate.valid
  assert certificate.observed_min_sep is None or certificate.observed_min_sep >= 3


def test_theoretical_and_observed_max_participations_for_n10_b3():
  spec = ParticipationSpec(horizon=10, min_sep=3, max_participations=None)
  selector = make_fixed_cycle_selection(horizon=10, min_sep=3, adjacency="replace_one")
  certificate = certify_participation_schedule(list(selector.batch_iterator(12, rng=7)), spec)
  assert theoretical_max_participations(spec) == 4
  assert certificate.observed_max_participations <= 4


def test_invalid_min_sep_schedule_fails_fast():
  spec = ParticipationSpec(horizon=2, min_sep=2, max_participations=None)
  with pytest.raises(ValueError, match="min_sep"):
    certify_participation_schedule([np.array([0]), np.array([0])], spec)


def test_max_participations_violation_fails_fast():
  spec = ParticipationSpec(horizon=3, min_sep=1, max_participations=2)
  with pytest.raises(ValueError, match="max_participations"):
    certify_participation_schedule([np.array([0]), np.array([0]), np.array([0])], spec)


@pytest.mark.parametrize("batches", [[np.array([0]), np.array([1])], [np.array([0])] * 4])
def test_horizon_mismatch_fails_fast(batches):
  spec = ParticipationSpec(horizon=3, min_sep=1, max_participations=None)
  with pytest.raises(ValueError, match="number of batches"):
    certify_participation_schedule(batches, spec)


def test_duplicate_records_within_a_batch_fail_fast():
  spec = ParticipationSpec(horizon=1, min_sep=1, max_participations=None)
  with pytest.raises(ValueError, match="at most once"):
    certify_participation_schedule([np.array([3, 3, 7])], spec)


def test_padding_is_ignored():
  spec = ParticipationSpec(horizon=2, min_sep=2, max_participations=1)
  certificate = certify_participation_schedule(
      [np.array([-1, -1, 3]), np.array([-1, -1, 4])], spec
  )
  assert certificate.valid
  assert certificate.observed_max_participations == 1
  assert certificate.observed_min_sep is None


def _strategy() -> BandInvMFStrategy:
  return BandInvMFStrategy(
      horizon=10,
      bandwidth=3,
      min_sep=3,
      max_participations=4,
      workload_coef=jnp.ones(10),
      noising_coef=jnp.ones(3),
      strategy_coef=jnp.ones(10),
      sensitivity_squared=jnp.array(1.0),
      objective=jnp.array(1.0),
  )


def test_strategy_spec_consistency():
  strategy = _strategy()
  matching = ParticipationSpec(horizon=10, min_sep=3, max_participations=4)
  assert participation_spec_from_strategy(strategy) == matching
  validate_participation_spec_against_strategy(matching, strategy)
  with pytest.raises(ValueError, match="exactly match"):
    validate_participation_spec_against_strategy(
        ParticipationSpec(horizon=10, min_sep=2, max_participations=4), strategy
    )


def test_adjacency_selects_expected_partition_type():
  add_remove = make_fixed_cycle_selection(horizon=10, min_sep=3, adjacency="add_remove")
  replace_one = make_fixed_cycle_selection(horizon=10, min_sep=3, adjacency="replace_one")
  assert add_remove.sampling_prob == 1.0
  assert add_remove.iterations == 10
  assert add_remove.cycle_length == 3
  assert add_remove.partition_type is batch_selection.PartitionType.INDEPENDENT
  assert replace_one.partition_type is batch_selection.PartitionType.EQUAL_SPLIT
  with pytest.raises(ValueError, match="adjacency"):
    make_fixed_cycle_selection(horizon=10, min_sep=3, adjacency="unknown")

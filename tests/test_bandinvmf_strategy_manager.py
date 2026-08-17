"""Regression tests for the BandInvMF strategy manager.

Guards the historical Nesterov path (``BandInvMFFitRequest``) against the new
prefix-sum support (``PrefixSumBandInvMFFitRequest``) that was wired in for the
correlated BandInvMF DP-AdamW baseline.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dp_muon.bandinvmf import BandInvMFStrategy, save_bandinv_strategy
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.training import bandinvmf_strategy_manager as strategies


def _nesterov_request(**changes):
  values = dict(
      horizon=4, min_sep=1, max_participations=2, bandwidth=2, momentum=0.9,
      learning_rate=0.1, reduction="mean", max_optimizer_steps=1,
      strategy_dir="/tmp/strategies", force_refit=False,
  )
  values.update(changes)
  return strategies.BandInvMFFitRequest(**values)


def _prefix_sum_request(**changes):
  values = dict(
      horizon=4, min_sep=1, max_participations=2, bandwidth=2,
      reduction="mean", max_optimizer_steps=1,
      strategy_dir="/tmp/strategies", force_refit=False,
  )
  values.update(changes)
  return strategies.PrefixSumBandInvMFFitRequest(**values)


def _strategy(request, *, workload_coef=None):
  if workload_coef is None:
    workload_coef = fixed_lr_nesterov_trajectory_workload_coef(
        request.horizon, getattr(request, "momentum", 0.9),
        getattr(request, "learning_rate", 0.1),
    )
  return BandInvMFStrategy(
      horizon=request.horizon, bandwidth=request.bandwidth, min_sep=request.min_sep,
      max_participations=request.max_participations,
      workload_coef=np.asarray(workload_coef),
      noising_coef=jnp.ones((request.bandwidth,)),
      strategy_coef=jnp.ones((request.horizon,)),
      sensitivity_squared=jnp.asarray(1.0), objective=jnp.asarray(1.0),
  )


def test_bandinv_mf_fit_request_is_importable_and_constructable():
  request = _nesterov_request()
  assert request.horizon == 4
  assert request.momentum == 0.9
  assert request.learning_rate == 0.1
  assert request.force_refit is False
  assert strategies.BandInvMFFitRequest is not strategies.PrefixSumBandInvMFFitRequest


def test_prefix_sum_request_has_no_momentum_or_learning_rate():
  request = _prefix_sum_request()
  assert not hasattr(request, "momentum")
  assert not hasattr(request, "learning_rate")


def test_prefix_sum_strategy_compatibility_requires_all_ones_workload():
  request = _prefix_sum_request()
  compatible = _strategy(request, workload_coef=np.ones(request.horizon))
  assert strategies._prefix_sum_strategy_is_compatible(compatible, request)
  incomplete = _strategy(request, workload_coef=np.full(request.horizon, 2.0))
  assert not strategies._prefix_sum_strategy_is_compatible(incomplete, request)


def test_prefix_sum_wrong_workload_refits_even_with_prefix_sum_metadata(tmp_path):
  request = _prefix_sum_request(strategy_dir=tmp_path)
  path = strategies.prefix_sum_strategy_artifact_path(
      request.strategy_dir, horizon=request.horizon, min_sep=request.min_sep,
      max_participations=request.max_participations, bandwidth=request.bandwidth,
      reduction=request.reduction, max_optimizer_steps=request.max_optimizer_steps,
  )
  # Metadata claims prefix-sum but the workload is not all-ones: must not hit.
  wrong_workload = _strategy(request, workload_coef=np.full(request.horizon, 3.0))
  save_bandinv_strategy(
      path, wrong_workload, reduction="mean", workload_type="prefix-sum",
      momentum=None, learning_rate=None, max_optimizer_steps=1,
  )
  assert strategies._load_compatible_prefix_sum_snapshot_unlocked(path, request) is None
  calls = []
  _, action = strategies.get_or_fit_prefix_sum_strategy_snapshot(
      request,
      fit_strategy=lambda *args, **kwargs: calls.append(1) or _strategy(
          request, workload_coef=np.ones(request.horizon)
      ),
  )
  assert action == "fit"
  assert calls == [1]


def test_nesterov_get_or_fit_publishes_reusable_snapshot(tmp_path):
  request = _nesterov_request(strategy_dir=tmp_path)
  snapshot, action = strategies.get_or_fit_strategy_snapshot(
      request, fit_strategy=lambda *args, **kwargs: _strategy(request)
  )
  assert action == "fit"
  snapshot2, action2 = strategies.get_or_fit_strategy_snapshot(
      request,
      fit_strategy=lambda *args, **kwargs: pytest.fail("compatible Nesterov artifact must be reused"),
  )
  assert action2 == "reuse"
  assert snapshot.path == snapshot2.path
  assert snapshot.sha256 == snapshot2.sha256


def test_nesterov_and_prefix_sum_artifact_paths_do_not_collide(tmp_path):
  nesterov = strategies.strategy_artifact_path(
      tmp_path, horizon=4, min_sep=1, max_participations=2, bandwidth=2,
      momentum=0.9, learning_rate=0.1, reduction="mean", max_optimizer_steps=1,
  )
  prefix_sum = strategies.prefix_sum_strategy_artifact_path(
      tmp_path, horizon=4, min_sep=1, max_participations=2, bandwidth=2,
      reduction="mean", max_optimizer_steps=1,
  )
  assert nesterov != prefix_sum
  assert "nesterov-trajectory" in nesterov.name
  assert "prefix-sum" in prefix_sum.name
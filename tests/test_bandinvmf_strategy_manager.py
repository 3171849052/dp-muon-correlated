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
from dp_muon.optim import (
    adam_first_moment_workload_matrix,
    fixed_lr_nesterov_trajectory_workload_coef,
)
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


def test_decayed_prefix_request_records_learning_rate_and_weight_decay():
  request = _prefix_sum_request()
  assert not hasattr(request, "momentum")
  assert request.learning_rate == 1.0
  assert request.weight_decay == 0.0


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
      path, wrong_workload, reduction="mean", workload_type="decayed-prefix-sum",
      momentum=None, learning_rate=request.learning_rate,
      weight_decay=request.weight_decay, max_optimizer_steps=1,
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
  assert "nesterov-decayed-trajectory" in nesterov.name
  assert "decayed-prefix-sum" in prefix_sum.name


def test_changed_weight_decay_cannot_reuse_prefix_artifact(tmp_path):
  request = _prefix_sum_request(
      strategy_dir=tmp_path, learning_rate=0.1, weight_decay=0.1
  )
  snapshot, action = strategies.get_or_fit_prefix_sum_strategy_snapshot(request)
  assert action == "fit"
  changed = _prefix_sum_request(
      strategy_dir=tmp_path, learning_rate=0.1, weight_decay=0.2
  )
  assert strategies.prefix_sum_strategy_artifact_path(
      request.strategy_dir, horizon=request.horizon, min_sep=request.min_sep,
      max_participations=request.max_participations, bandwidth=request.bandwidth,
      learning_rate=request.learning_rate, weight_decay=request.weight_decay,
      reduction=request.reduction, max_optimizer_steps=request.max_optimizer_steps,
  ) != strategies.prefix_sum_strategy_artifact_path(
      changed.strategy_dir, horizon=changed.horizon, min_sep=changed.min_sep,
      max_participations=changed.max_participations, bandwidth=changed.bandwidth,
      learning_rate=changed.learning_rate, weight_decay=changed.weight_decay,
      reduction=changed.reduction, max_optimizer_steps=changed.max_optimizer_steps,
  )
  assert strategies._load_compatible_prefix_sum_snapshot_unlocked(
      snapshot.path, changed
  ) is None


def test_missing_weight_decay_metadata_is_not_a_cache_hit(tmp_path):
  request = _prefix_sum_request(strategy_dir=tmp_path)
  path = strategies.prefix_sum_strategy_artifact_path(
      request.strategy_dir, horizon=request.horizon, min_sep=request.min_sep,
      max_participations=request.max_participations, bandwidth=request.bandwidth,
      learning_rate=request.learning_rate, weight_decay=request.weight_decay,
      reduction=request.reduction, max_optimizer_steps=request.max_optimizer_steps,
  )
  strategy = _strategy(request, workload_coef=np.ones(request.horizon))
  np.savez(
      path, horizon=np.asarray(strategy.horizon), bandwidth=np.asarray(strategy.bandwidth),
      min_sep=np.asarray(strategy.min_sep), max_participations=np.asarray(strategy.max_participations),
      workload_coef=np.asarray(strategy.workload_coef), noising_coef=np.asarray(strategy.noising_coef),
      strategy_coef=np.asarray(strategy.strategy_coef), sensitivity_squared=np.asarray(strategy.sensitivity_squared),
      objective=np.asarray(strategy.objective), reduction=np.asarray("mean"),
      workload_type=np.asarray("decayed-prefix-sum"), momentum=np.asarray(np.nan),
      learning_rate=np.asarray(request.learning_rate),
      max_optimizer_steps=np.asarray(request.max_optimizer_steps),
  )
  assert strategies._load_compatible_prefix_sum_snapshot_unlocked(path, request) is None


def test_global_correlated_request_uses_full_adamw_workload_and_reuses(
    tmp_path,
):
  request = strategies.GlobalCorrelatedBandInvMFFitRequest(
      horizon=6, min_sep=1, max_participations=2, bandwidth=2,
      beta1=.8, learning_rate=.05, weight_decay=.01,
      reduction="mean", max_optimizer_steps=1, strategy_dir=tmp_path,
      force_refit=False,
  )
  captured = {}

  def fake_fit(horizon, bandwidth, min_sep, **kwargs):
    captured.update(horizon=horizon, bandwidth=bandwidth, min_sep=min_sep, **kwargs)
    workload = np.asarray(kwargs["workload_matrix"])
    return BandInvMFStrategy(
        horizon=horizon, bandwidth=bandwidth, min_sep=min_sep,
        max_participations=request.max_participations, workload_coef=None,
        workload_matrix=jnp.asarray(workload),
        noising_coef=jnp.asarray([1.0, .2]),
        strategy_coef=jnp.ones((horizon,)), sensitivity_squared=jnp.asarray(1.0),
        objective=jnp.asarray(1.0),
    )

  snapshot, action = strategies.get_or_fit_global_correlated_strategy_snapshot(
      request, fit_strategy=fake_fit
  )
  assert action == "fit"
  assert captured["horizon"] == request.horizon
  np.testing.assert_allclose(
      captured["workload_matrix"],
      np.asarray(adam_first_moment_workload_matrix(
          request.horizon, request.beta1, request.learning_rate,
          request.weight_decay,
      )),
  )
  assert "b10.8" in snapshot.path.name
  assert "tau" not in snapshot.path.name

  snapshot2, action2 = strategies.get_or_fit_global_correlated_strategy_snapshot(
      request,
      fit_strategy=lambda *args, **kwargs: pytest.fail("global strategy must reuse"),
  )
  assert action2 == "reuse"
  assert snapshot.path == snapshot2.path


def test_global_correlated_workload_changes_with_beta1(tmp_path):
  base = dict(
      horizon=5, min_sep=1, max_participations=2, bandwidth=2,
      learning_rate=.05, weight_decay=.01, reduction="mean",
      max_optimizer_steps=1, strategy_dir=tmp_path, force_refit=False,
  )
  first = strategies.GlobalCorrelatedBandInvMFFitRequest(beta1=.2, **base)
  second = strategies.GlobalCorrelatedBandInvMFFitRequest(beta1=.8, **base)
  assert not np.allclose(
      np.asarray(adam_first_moment_workload_matrix(
          first.horizon, first.beta1, first.learning_rate, first.weight_decay
      )),
      np.asarray(adam_first_moment_workload_matrix(
          second.horizon, second.beta1, second.learning_rate, second.weight_decay
      )),
  )
  assert strategies.global_correlated_strategy_artifact_path(
      tmp_path, horizon=first.horizon, min_sep=first.min_sep,
      max_participations=first.max_participations, bandwidth=first.bandwidth,
      beta1=first.beta1, learning_rate=first.learning_rate,
      weight_decay=first.weight_decay, reduction=first.reduction,
      max_optimizer_steps=first.max_optimizer_steps,
  ) != strategies.global_correlated_strategy_artifact_path(
      tmp_path, horizon=second.horizon, min_sep=second.min_sep,
      max_participations=second.max_participations, bandwidth=second.bandwidth,
      beta1=second.beta1, learning_rate=second.learning_rate,
      weight_decay=second.weight_decay, reduction=second.reduction,
      max_optimizer_steps=second.max_optimizer_steps,
  )

"""Algorithm, pairing, diagnostic, and smoke tests for Experiment 7b."""

import csv
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from exp7.core import init_exp7_train_state, make_exp7_train_step
from exp7b.core import (
    gamma_prime_from_ratio, make_exp7b_train_step, paper_bc_preconditioner,
    phi_infinity,
)
from exp7b.diagnostics import (
    NORM_NAMES, aggregate_window_rows, histogram_quantile, stage_metrics, window_row,
)
from exp7b.online_shadow import Exp7bWindowCollector
from exp7b.run import run_smoke


def _strategy(horizon=4, coef=(1.0, -0.25)):
  noising = jnp.asarray(coef, jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=noising, min_sep=1, max_participations=1
  )
  return BandInvMFStrategy(
      horizon=horizon, bandwidth=len(coef), min_sep=1, max_participations=1,
      workload_coef=jnp.ones(horizon), noising_coef=noising,
      strategy_coef=toeplitz.inverse_coef(noising, horizon),
      sensitivity_squared=sensitivity, objective=jnp.asarray(0.0),
  )


def _calibration(strategy):
  return calibrate_nonamplified_bandinv(
      epsilon=2.0, delta=1e-5, clip_norm=10.0, normalize_by=2.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )


def _loss(params, batch):
  prediction = jnp.dot(params["w"], batch["x"][0])
  return .5 * (prediction - batch["y"][0]) ** 2


def _batch(step=1):
  return {
      "x": jnp.asarray([[1.0, step / 10], [1.0, -step / 10]], jnp.float32),
      "y": jnp.asarray([.5, -.25], jnp.float32),
  }


def _assert_tree_allclose(left, right, **kwargs):
  for a, b in zip(
      jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
  ):
    if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
    else:
      np.testing.assert_allclose(a, b, **kwargs)


def _steps(strategy, calibration, algorithm, key, gamma_prime, horizon=4):
  participation = ParticipationSpec(horizon, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  state = init_exp7_train_state(params, strategy, key)
  step_fn = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, algorithm=algorithm,
      learning_rate=.02, beta1=.9, beta2=.8, eps=1e-3,
      weight_decay=.01, gamma_prime=gamma_prime,
  ))
  output = []
  for index in range(horizon):
    state = step_fn(state, _batch(index + 1))
    output.append(state)
  return output


def test_gamma_prime_is_ratio_times_phi_infinity():
  strategy = _strategy(coef=(1.0, -.5, .25))
  sigma, ratio = .7, 1.25
  expected_phi = sigma ** 2 * (1.0 + .25 + .0625)
  phi_inf, gamma_prime = gamma_prime_from_ratio(strategy, sigma, ratio)
  assert float(phi_infinity(strategy, sigma)) == pytest.approx(expected_phi)
  assert float(phi_inf) == pytest.approx(expected_phi)
  assert float(gamma_prime) == pytest.approx(ratio * expected_phi)
  with pytest.raises(ValueError):
    gamma_prime_from_ratio(strategy, sigma, 0.0)


def test_correlated_phi_ema_is_identical_to_exp7():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  participation = ParticipationSpec(horizon, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  key = jax.random.key(7)
  old_state = init_exp7_train_state(params, strategy, key)
  new_state = init_exp7_train_state(params, strategy, key)
  old_step = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, algorithm="bc", learning_rate=.02,
      beta2=.8,
  ))
  _, gamma_prime = gamma_prime_from_ratio(strategy, calibration.iid_noise_std)
  new_step = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, algorithm="bc", learning_rate=.02,
      beta2=.8, gamma_prime=gamma_prime,
  ))
  for index in range(horizon):
    batch = _batch(index + 1)
    old_state, new_state = old_step(old_state, batch), new_step(new_state, batch)
    np.testing.assert_array_equal(old_state.phi_t, new_state.phi_t)
    np.testing.assert_array_equal(old_state.bias_v, new_state.bias_v)


def test_paper_preconditioner_is_exact_and_has_no_outer_adam_epsilon():
  corrected = jnp.asarray([-3.0, 0.25, 4.0], jnp.float32)
  gamma_prime = 1.0
  expected = 1.0 / np.sqrt(np.maximum(np.asarray(corrected), gamma_prime))
  actual = paper_bc_preconditioner(corrected, gamma_prime)
  np.testing.assert_array_equal(actual, expected.astype(np.float32))
  assert "+ eps" not in inspect.getsource(paper_bc_preconditioner)
  assert float(actual[0]) == pytest.approx(1.0 / np.sqrt(gamma_prime))
  assert np.all(np.asarray(actual) <= 1.0 / np.sqrt(gamma_prime))


def test_bc_parameter_update_uses_paper_preconditioner_exactly():
  strategy = _strategy(horizon=1, coef=(1.0,))
  calibration = _calibration(strategy)
  participation = ParticipationSpec(1, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  gamma_prime = .75
  beta1, beta2, lr, wd = .9, .8, .02, .01
  state = init_exp7_train_state(params, strategy, jax.random.key(11))
  step_fn = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, algorithm="bc",
      learning_rate=lr, beta1=beta1, beta2=beta2, eps=.25,
      weight_decay=wd, gamma_prime=gamma_prime,
  ))
  updated = step_fn(state, _batch())
  t = updated.step
  mhat = jax.tree_util.tree_map(lambda x: x / (1 - beta1 ** t), updated.dp_m)
  vhat = jax.tree_util.tree_map(lambda x: x / (1 - beta2 ** t), updated.v11)
  phi_hat = updated.bias_v / (1 - beta2 ** t)
  expected = jax.tree_util.tree_map(
      lambda old, m, v: (1 - lr * wd) * old
      - lr * m / jnp.sqrt(jnp.maximum(v - phi_hat, gamma_prime)),
      params, mhat, vhat,
  )
  _assert_tree_allclose(updated.params, expected, rtol=1e-7, atol=1e-7)


def test_floor_activation_fraction_matches_coordinate_definition():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  _, gamma_prime = gamma_prime_from_ratio(strategy, calibration.iid_noise_std)
  states = _steps(
      strategy, calibration, "bc", jax.random.key(3), float(gamma_prime), horizon
  )
  collector = Exp7bWindowCollector(
      {"w": jnp.asarray([.1, -.2], jnp.float32)}, seed=3, algorithm="bc",
      beta1=.9, beta2=.8, learning_rate=.02, weight_decay=.01, eps=1e-3,
      gamma_prime=float(gamma_prime), window_size=horizon, histogram_bins=64,
  )
  expected_nonpositive, expected_floor = [], []
  for index, state in enumerate(states, 1):
    collector.after_step(state, index)
    vhat = state.v11["w"] / (1 - .8 ** state.step)
    corrected = vhat - state.bias_v / (1 - .8 ** state.step)
    expected_nonpositive.append(float(jnp.mean(corrected <= 0)))
    expected_floor.append(float(jnp.mean(corrected <= gamma_prime)))
  row = collector.finalize()[0]
  assert row["corrected_v_nonpositive_fraction"] == pytest.approx(
      np.mean(expected_nonpositive)
  )
  assert row["floor_activation_fraction"] == pytest.approx(np.mean(expected_floor))
  assert row["p_bc_max"] <= 1.0 / np.sqrt(float(gamma_prime)) * (1 + 1e-6)


def test_baseline_trajectory_is_identical_to_exp7():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  participation = ParticipationSpec(horizon, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  key = jax.random.key(123)
  old_state = init_exp7_train_state(params, strategy, key)
  new_state = init_exp7_train_state(params, strategy, key)
  kwargs = dict(
      algorithm="baseline", learning_rate=.02, beta1=.9, beta2=.999,
      eps=1e-8, weight_decay=.01,
  )
  old_step = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, **kwargs
  ))
  new_step = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, gamma_prime=1.0, **kwargs
  ))
  for index in range(horizon):
    batch = _batch(index + 1)
    old_state, new_state = old_step(old_state, batch), new_step(new_state, batch)
    _assert_tree_allclose(old_state, new_state, rtol=0, atol=0)


def test_paired_branches_share_initialization_batches_and_latent_noise():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  _, gamma_prime = gamma_prime_from_ratio(strategy, calibration.iid_noise_std)
  key = jax.random.key(99)
  baseline = _steps(strategy, calibration, "baseline", key, float(gamma_prime), horizon)
  bc = _steps(strategy, calibration, "bc", key, float(gamma_prime), horizon)
  for baseline_state, bc_state in zip(baseline, bc, strict=True):
    _assert_tree_allclose(
        baseline_state.last_latent_noise, bc_state.last_latent_noise, rtol=0, atol=0
    )
    _assert_tree_allclose(
        baseline_state.last_noise, bc_state.last_noise, rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        jax.random.key_data(baseline_state.rng_key),
        jax.random.key_data(bc_state.rng_key),
    )


def _synthetic_histogram(values, *, bins, implied_p_max):
  values = np.asarray(values, np.float64)
  indices = np.minimum(
      np.floor(values / implied_p_max * bins).astype(np.int64), bins - 1
  )
  return np.bincount(indices, minlength=bins)


def _cancellation_windows(total_steps=488, window_size=16):
  rows = []
  for index, start in enumerate(range(1, total_steps + 1, window_size)):
    end = min(total_steps, start + window_size - 1)
    rows.append({
        "seed": 0, "algorithm": "bc", "window_index": index,
        "start_step": start, "end_step": end,
        "C_00": 1.0, "C_10": 2.0, "C_01": 3.0, "C_11": 5.0,
        "C_BC": 1.5, "C_real": 1.5,
        "C_dynamic_clean_p": 1.0, "gap": .5,
    })
  return rows


def _step_record(step, samples, norm_value, *, bins=8, implied_p_max=8.0):
  return {
      "seed": 0, "algorithm": "bc", "step": step,
      **{f"nonpositive_v_fraction_{path}": step / 1000 for path in (
          "00", "10", "01", "11", "BC"
      )},
      "corrected_v_nonpositive_fraction": step / 1000,
      "floor_activation_fraction": step / 2000,
      "p_bc_histogram": _synthetic_histogram(
          samples, bins=bins, implied_p_max=implied_p_max
      ),
      "p_bc_max": max(samples),
      "raw_optimizer_update_l2": norm_value,
      "applied_parameter_update_l2": 2 * norm_value,
      "parameter_l2": 3 * norm_value,
  }


def test_stage_stability_uses_exact_steps_at_97_98_boundary():
  implied_p_max, bins = 8.0, 8
  samples_by_step = {
      step: np.asarray([step % 7 + .1, (step * 3) % 7 + .2])
      for step in range(1, 489)
  }
  norm_values = np.arange(1, 489, dtype=np.float64)
  norm_values[97] = 10_000.0  # step 98: must never leak into early stage.
  records = [
      _step_record(step, samples_by_step[step], norm_values[step - 1],
                   bins=bins, implied_p_max=implied_p_max)
      for step in range(1, 489)
  ]
  rows = _cancellation_windows()
  early = stage_metrics(
      rows, stage_start=1, stage_end=97, step_diagnostics=records,
      implied_p_max=implied_p_max,
  )
  late = stage_metrics(
      rows, stage_start=98, stage_end=488, step_diagnostics=records,
      implied_p_max=implied_p_max,
  )

  assert early["covered_stability_steps"] == 97
  assert late["covered_stability_steps"] == 391
  assert early["raw_optimizer_update_l2_max"] == pytest.approx(97.0)
  assert late["raw_optimizer_update_l2_max"] == pytest.approx(10_000.0)
  assert early["raw_optimizer_update_l2_min"] == pytest.approx(1.0)
  assert late["raw_optimizer_update_l2_min"] == pytest.approx(99.0)
  assert early["raw_optimizer_update_l2_std"] == pytest.approx(
      np.std(norm_values[:97])
  )
  assert late["raw_optimizer_update_l2_std"] == pytest.approx(
      np.std(norm_values[97:])
  )
  # Window 7 is 97-112, but early receives exactly its first step.
  overlapping_window_max_average = np.mean([16, 32, 48, 64, 80, 96, 10_000])
  assert early["raw_optimizer_update_l2_max"] != pytest.approx(
      overlapping_window_max_average
  )

  for summary, selected_steps in (
      (early, range(1, 98)), (late, range(98, 489))
  ):
    direct_samples = np.concatenate([samples_by_step[step] for step in selected_steps])
    direct_histogram = _synthetic_histogram(
        direct_samples, bins=bins, implied_p_max=implied_p_max
    )
    assert summary["p_bc_median"] == pytest.approx(
        histogram_quantile(direct_histogram, .5, implied_p_max)
    )
    assert summary["p_bc_q99"] == pytest.approx(
        histogram_quantile(direct_histogram, .99, implied_p_max)
    )
    assert summary["p_bc_q99_9"] == pytest.approx(
        histogram_quantile(direct_histogram, .999, implied_p_max)
    )


def _window_stability(*, mean, std, minimum, maximum, p_quantile, p_max):
  stability = {
      "corrected_v_nonpositive_fraction": .25,
      "floor_activation_fraction": .5,
      "p_bc_median": p_quantile,
      "p_bc_q99": p_quantile + 1,
      "p_bc_q99_9": p_quantile + 2,
      "p_bc_max": p_max,
  }
  for name in NORM_NAMES:
    stability.update({
        f"{name}_mean": mean, f"{name}_std": std,
        f"{name}_min": minimum, f"{name}_max": maximum,
    })
  return stability


def test_cross_seed_window_summary_pools_std_and_preserves_extrema_names():
  scores = {"00": 1., "10": 2., "01": 3., "11": 5., "BC": 1.5}
  negative = {path: .1 for path in ("00", "10", "01", "11", "BC")}
  rows = [
      window_row(
          seed=0, algorithm="bc", window_index=0, start_step=1, end_step=2,
          mean_p_relative_change=.1, scores=scores, negative_fractions=negative,
          stability=_window_stability(
              mean=2., std=1., minimum=1., maximum=3., p_quantile=2., p_max=5.
          ),
      ),
      window_row(
          seed=1, algorithm="bc", window_index=0, start_step=1, end_step=2,
          mean_p_relative_change=.2, scores=scores, negative_fractions=negative,
          stability=_window_stability(
              mean=10., std=2., minimum=7., maximum=13., p_quantile=6., p_max=14.
          ),
      ),
  ]
  summary = aggregate_window_rows(rows)[0]
  direct_values = np.asarray([1., 3., 8., 12.])
  assert summary["raw_optimizer_update_l2_mean"] == pytest.approx(
      np.mean(direct_values)
  )
  assert summary["raw_optimizer_update_l2_std"] == pytest.approx(
      np.std(direct_values)
  )
  assert summary["raw_optimizer_update_l2_min"] == pytest.approx(1.)
  assert summary["raw_optimizer_update_l2_max"] == pytest.approx(13.)
  assert summary["p_bc_max"] == pytest.approx(14.)
  assert summary["mean_window_p_bc_median"] == pytest.approx(4.)
  assert "p_bc_median_mean" not in summary


def test_small_smoke_writes_all_required_outputs(tmp_path: Path):
  run_smoke(tmp_path, [0], gamma_prime_ratio=1.0)
  required = (
      "window_diagnostics_baseline_seed0.csv",
      "window_diagnostics_bc_seed0.csv",
      "window_summary.csv", "summary.json", "shadow_cancellation.png",
      "bc_preconditioner_floor_activation.png", "update_norm_diagnostics.png",
  )
  for name in required:
    assert (tmp_path / name).is_file(), name
  summary = json.loads((tmp_path / "summary.json").read_text())
  assert summary["smoke"] is True
  assert summary["gamma_prime_ratio"] == 1.0
  assert summary["gamma_prime"] == pytest.approx(summary["phi_infty"])
  assert summary["implied_p_max"] == pytest.approx(1 / np.sqrt(summary["gamma_prime"]))
  assert set(summary["per_seed"]["0"]) == {"baseline", "bc"}
  assert set(summary["per_seed"]["0"]["bc"]) >= {
      "final_test_loss", "final_test_accuracy",
      "early_steps_1_97", "late_steps_98_488",
  }
  assert summary["per_seed"]["0"]["bc"]["early_steps_1_97"][
      "covered_stability_steps"
  ] == 20
  with (tmp_path / "window_diagnostics_bc_seed0.csv").open(newline="") as stream:
    row = next(csv.DictReader(stream))
  assert set(row) >= {
      "corrected_v_nonpositive_fraction", "floor_activation_fraction",
      "p_bc_median", "p_bc_q99", "p_bc_q99_9", "p_bc_max",
      "raw_optimizer_update_l2_mean", "applied_parameter_update_l2_mean",
      "parameter_l2_mean",
  }

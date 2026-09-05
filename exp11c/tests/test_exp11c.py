"""Tests for Exp11c's exact matrix capture and scale-blindness reduction."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.optim import (
    PreQMatrixState,
    extract_pre_q_matrix,
    make_pre_q_matrix_hook,
    muon_transform,
)
from dp_muon.privacy import calibrate_nonamplified_iid
from exp11b.run import (
    DPMuonSettings,
    TARGET_PARAMETER_PATHS,
    _smoke_params,
)
from exp11c.plotting import (
    PAIR_NAMES,
    TRAJECTORIES,
    load_scale_blindness,
    plot_scale_blindness,
    save_scale_blindness_csv,
)
from exp11c.run import (
    _extract_target_pre_q_matrices,
    _pairwise_metrics,
    _run_trajectory_group,
    ideal_muon_q,
    run_smoke,
)


def _calibration(noise_std=.1):
  value = calibrate_nonamplified_iid(
      epsilon=3, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", max_participations=2,
  )
  return replace(value, iid_noise_std=noise_std)


def _settings():
  return DPMuonSettings(
      muon_learning_rate=.01, muon_weight_decay=0.0, momentum=.8,
      ns_steps=2, consistent_rms=.2, adamw_learning_rate=.01,
      adamw_beta1=.9, adamw_beta2=.99, adamw_eps=1e-6,
      adamw_weight_decay=0.0, microbatch_size=None, use_bf16_ns=False,
  )


def test_matrix_hook_is_after_nesterov_and_before_q():
  beta = .5
  first = jnp.asarray([[2., 0.], [0., 1.]], jnp.float32)
  second = jnp.asarray([[0., 3.], [4., 0.]], jnp.float32)
  transform = muon_transform(
      learning_rate=.1, weight_decay=0.0, momentum=beta, ns_steps=2,
      consistent_rms=.2, use_bf16_ns=False,
      pre_q_hook=make_pre_q_matrix_hook(("w",)),
  )
  state = transform.init({"w": first})
  _, state = transform.update({"w": first}, state, {"w": first})
  _, state = transform.update({"w": second}, state, {"w": first})
  old_momentum = (1.0 - beta) * first
  new_momentum = beta * old_momentum + (1.0 - beta) * second
  expected = (1.0 - beta) * second + beta * new_momentum
  np.testing.assert_array_equal(np.asarray(extract_pre_q_matrix(state)), np.asarray(expected))
  final_update, _ = transform.update({"w": second}, state, {"w": first})
  assert not np.allclose(np.asarray(final_update["w"]), np.asarray(expected))


def test_matrix_hook_forwards_updates_and_keeps_only_latest_matrix():
  hook = make_pre_q_matrix_hook(("w",))
  params = {"w": jnp.eye(2, dtype=jnp.float32)}
  state = hook.init(params)
  first = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
  second = {"w": jnp.full((2, 2), 3, dtype=jnp.float32)}
  update, state = hook.update(first, state, params)
  assert isinstance(state, PreQMatrixState)
  np.testing.assert_array_equal(update["w"], first["w"])
  _, state = hook.update(second, state, params)
  np.testing.assert_array_equal(extract_pre_q_matrix(state), second["w"])


def test_ideal_muon_q_uses_float64_polar_factor():
  matrix = np.asarray([[3.0, 1.0], [-2.0, 4.0]], dtype=np.float32)
  u, _, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
  np.testing.assert_array_equal(ideal_muon_q(matrix), u @ vh)
  np.testing.assert_allclose(ideal_muon_q(2.75 * matrix), ideal_muon_q(matrix))


def test_three_trajectories_capture_real_matrices_and_reduce_without_replay():
  params = _smoke_params()

  def loss_fn(parameters, batch):
    selected = sum(
        jnp.sum(parameters["blocks"][block]["attention"]["query"]["kernel"])
        for block in (0, 5, 11)
    )
    return selected * batch["scale"][0]

  result = _run_trajectory_group(
      initial_params=params,
      batches=[{"scale": jnp.asarray([1.0, 1.0], jnp.float32)},
               {"scale": jnp.asarray([.5, .5], jnp.float32)}],
      horizon=2,
      calibrations={3: _calibration(.2), 8: _calibration(.1)},
      loss_fn=loss_fn,
      settings=_settings(), seed=4, steps=(1, 2),
  )
  assert result.trajectories == TRAJECTORIES
  assert result.pair_names == PAIR_NAMES
  assert result.matrix_frobenius_norms.shape == (3, 2, 3)
  assert result.ideal_q_pairwise_cosines.shape == (2, 3, 3)
  assert np.all(result.matrix_frobenius_norms[1] != result.matrix_frobenius_norms[2])
  assert np.all(np.isfinite(result.ideal_q_pairwise_frobenius_distances))


def test_compact_artifact_and_plot(tmp_path):
  result = run_smoke(tmp_path)
  loaded = load_scale_blindness(tmp_path / "scale_blindness.npz")
  assert loaded["matrix_frobenius_norms"].shape == (3, 3, 3)
  csv_path = save_scale_blindness_csv(
      tmp_path / "scale_blindness.npz", tmp_path / "scale_blindness.csv"
  )
  assert csv_path.is_file()
  plot_path = plot_scale_blindness(
      tmp_path / "scale_blindness.npz", tmp_path / "scale_blindness-test.png"
  )
  assert plot_path.is_file() and plot_path.stat().st_size > 0
  assert result.steps.tolist() == [1, 2, 3]


def test_matrix_extractor_requires_all_three_target_states():
  state = {"one": PreQMatrixState(jnp.eye(2))}
  try:
    _extract_target_pre_q_matrices(state)
  except ValueError as error:
    assert "three" in str(error)
  else:
    raise AssertionError("incomplete matrix hook state must fail")


def test_pairwise_metrics_are_zero_for_identical_ideal_q():
  matrix = ideal_muon_q(np.asarray([[2., 0.], [0., 1.]], dtype=np.float32))
  distances, cosines = _pairwise_metrics(np.stack([matrix, matrix, matrix]))
  np.testing.assert_allclose(distances, 0.0)
  np.testing.assert_allclose(cosines, 1.0)

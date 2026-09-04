"""Tests for the Exp11b pairing, capture, artifacts, and smoke wiring."""

import csv
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.optim import (
    extract_pre_q_singular_values,
    make_pre_q_svd_hook,
    muon_transform,
)
from dp_muon.privacy import calibrate_nonamplified_iid
import dp_muon.training.nonamplified_dpmuon as dpmuon
from exp11b.plotting import (
    load_spectra,
    plot_singular_spectra,
    save_spectra,
    save_spectra_csv,
)
from exp11b.run import (
    DPMuonSettings,
    REQUIRED_STEPS,
    TARGET_EPSILONS,
    TARGET_LAYER_NAMES,
    TARGET_PARAMETER_PATHS,
    _calibrations,
    _extract_target_spectra,
    _run_trajectory_group,
    _smoke_params,
    run_smoke,
)


def _params():
  def dense(scale=1.0):
    return {
        "kernel": jnp.eye(2, dtype=jnp.float32) * scale,
        "bias": jnp.zeros((2,), dtype=jnp.float32),
    }
  blocks = []
  for block in range(12):
    blocks.append({
        "attention": {
            name: dense(1.0 + block * .01)
            for name in ("query", "key", "value", "out")
        },
        "mlp": {name: dense() for name in ("dense0", "dense1")},
    })
  return {"blocks": tuple(blocks), "head": dense()}


def _loss(params, batch):
  selected = sum(
      jnp.sum(params["blocks"][block]["attention"]["query"]["kernel"])
      for block in (0, 5, 11)
  )
  return (selected + jnp.sum(params["head"]["kernel"])) * batch["scale"][0]


def _calibration(epsilon=3, noise_std=.25):
  calibration = calibrate_nonamplified_iid(
      epsilon=epsilon,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=2.0,
      adjacency="add_remove",
      max_participations=2,
  )
  return replace(calibration, iid_noise_std=noise_std)


def _settings():
  return DPMuonSettings(
      muon_learning_rate=.01,
      muon_weight_decay=0.0,
      momentum=.8,
      ns_steps=2,
      consistent_rms=.2,
      adamw_learning_rate=.01,
      adamw_beta1=.9,
      adamw_beta2=.99,
      adamw_eps=1e-6,
      adamw_weight_decay=0.0,
      microbatch_size=None,
      use_bf16_ns=False,
  )


def _batch(value):
  return {"scale": jnp.asarray([value, -value], dtype=jnp.float32)}


def test_calibration_eps8_is_smaller_than_eps3():
  eps3 = calibrate_nonamplified_iid(
      epsilon=3, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", max_participations=2,
  )
  eps8 = calibrate_nonamplified_iid(
      epsilon=8, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", max_participations=2,
  )
  assert eps8.iid_noise_std < eps3.iid_noise_std


def test_clean_keeps_clipping_but_skips_noise(monkeypatch):
  query_outputs = []
  real_factory = dpmuon.make_clipped_gradient_query

  def query_factory(*args, **kwargs):
    query = real_factory(*args, **kwargs)

    def wrapped(*query_args, **query_kwargs):
      result = query(*query_args, **query_kwargs)
      query_outputs.append(result)
      return result

    return wrapped

  def unexpected_noise(*args, **kwargs):
    raise AssertionError("clean path must not sample Gaussian noise")

  monkeypatch.setattr(dpmuon, "make_clipped_gradient_query", query_factory)
  monkeypatch.setattr(dpmuon, "_sample_iid_gaussian_noise", unexpected_noise)
  step, optimizer = dpmuon.make_nonamplified_dpmuon_train_step(
      _loss,
      _calibration(noise_std=.25),
      muon_learning_rate=.01,
      muon_weight_decay=0.0,
      momentum=.8,
      ns_steps=2,
      consistent_rms=.2,
      adamw_learning_rate=.01,
      adamw_beta1=.9,
      adamw_beta2=.99,
      adamw_eps=1e-6,
      adamw_weight_decay=0.0,
      add_noise=False,
      pre_q_parameter_paths=TARGET_PARAMETER_PATHS,
  )
  state = dpmuon.init_nonamplified_dpmuon_state(_params(), jax.random.key(0), optimizer)
  step(state, _batch(10.0))
  assert len(query_outputs) == 1
  norm = jnp.sqrt(sum(
      jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(query_outputs[0])
  ))
  assert float(norm) <= .5 + 1e-5


def test_dp_path_samples_iid_noise_and_eps_scales_are_distinct(monkeypatch):
  calls = []
  real_sampler = dpmuon.sample_iid_gaussian_noise

  def fixed_noise(key, template, noise_std):
    noise, next_key = real_sampler(key, template, noise_std)
    calls.append((noise_std, key, next_key))
    return noise, next_key

  monkeypatch.setattr(dpmuon, "_sample_iid_gaussian_noise", fixed_noise)
  steps = []
  for epsilon, std in ((3, .25), (8, .1)):
    step, optimizer = dpmuon.make_nonamplified_dpmuon_train_step(
        _loss,
        _calibration(epsilon=epsilon, noise_std=std),
        muon_learning_rate=.01,
        muon_weight_decay=0.0,
        momentum=.8,
        ns_steps=2,
        consistent_rms=.2,
        adamw_learning_rate=.01,
        adamw_beta1=.9,
        adamw_beta2=.99,
        adamw_eps=1e-6,
        adamw_weight_decay=0.0,
        add_noise=True,
        pre_q_parameter_paths=TARGET_PARAMETER_PATHS,
    )
    state = dpmuon.init_nonamplified_dpmuon_state(_params(), jax.random.key(epsilon), optimizer)
    steps.append((step, state))
  for step, state in steps:
    step(state, _batch(1.0))
  assert len(calls) == 2
  np.testing.assert_allclose([float(value[0]) for value in calls], [.25, .1])
  assert all(not np.array_equal(jax.random.key_data(key), jax.random.key_data(next_key))
             for _, key, next_key in calls)


def test_pre_q_hook_is_after_nesterov_and_before_q():
  beta = .5
  first = jnp.asarray([[2., 0.], [0., 1.]], jnp.float32)
  second = jnp.asarray([[0., 3.], [4., 0.]], jnp.float32)
  transform = muon_transform(
      learning_rate=.1,
      weight_decay=0.0,
      momentum=beta,
      ns_steps=2,
      consistent_rms=.2,
      use_bf16_ns=False,
      pre_q_hook=make_pre_q_svd_hook(("w",)),
  )
  state = transform.init({"w": first})
  _, state = transform.update({"w": first}, state, {"w": first})
  _, state = transform.update({"w": second}, state, {"w": first})
  old_momentum = (1.0 - beta) * first
  new_momentum = beta * old_momentum + (1.0 - beta) * second
  pre_q = (1.0 - beta) * second + beta * new_momentum
  expected = np.linalg.svd(np.asarray(pre_q), compute_uv=False)
  actual = np.asarray(extract_pre_q_singular_values(state))
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
  final_update, _ = transform.update({"w": second}, state, {"w": first})
  assert not np.allclose(np.asarray(final_update["w"]), np.asarray(pre_q))


def test_all_three_target_layers_are_captured_with_matching_rank():
  calibration = _calibration(noise_std=.01)
  step, optimizer = dpmuon.make_nonamplified_dpmuon_train_step(
      _loss,
      calibration,
      muon_learning_rate=.01,
      muon_weight_decay=0.0,
      momentum=.8,
      ns_steps=2,
      consistent_rms=.2,
      adamw_learning_rate=.01,
      adamw_beta1=.9,
      adamw_beta2=.99,
      adamw_eps=1e-6,
      adamw_weight_decay=0.0,
      add_noise=False,
      pre_q_parameter_paths=TARGET_PARAMETER_PATHS,
  )
  state = dpmuon.init_nonamplified_dpmuon_state(_params(), jax.random.key(0), optimizer)
  state = step(state, _batch(1.0))
  spectra = _extract_target_spectra(state.optimizer_state)
  assert spectra.shape == (3, 2)
  assert np.all(np.diff(spectra, axis=-1) <= 0)


def test_three_trajectories_share_batches_and_clean_is_repeated():
  seen = []

  class RecordingBatches:
    def __iter__(self):
      for value in (1.0, 2.0, 3.0):
        seen.append(value)
        yield _batch(value)

  calibrations = {epsilon: _calibration(epsilon=epsilon, noise_std=.02 if epsilon == 3 else .01)
                  for epsilon in TARGET_EPSILONS}
  result = _run_trajectory_group(
      initial_params=_params(),
      batches=RecordingBatches(),
      horizon=3,
      calibrations=calibrations,
      loss_fn=_loss,
      settings=_settings(),
      seed=4,
      steps=(1, 3),
  )
  assert seen == [1.0, 2.0, 3.0]
  assert result.steps.tolist() == [1, 3]
  assert result.layers == TARGET_LAYER_NAMES
  assert result.clean_singular_values.shape == (2, 2, 3, 2)
  assert result.dp_singular_values.shape == (2, 2, 3, 2)
  np.testing.assert_array_equal(
      result.clean_singular_values[0], result.clean_singular_values[1]
  )
  assert not np.array_equal(result.dp_singular_values[0], result.dp_singular_values[1])


def test_calibration_helper_uses_required_clip_and_delta():
  class Config:
    clip_norm = 1.0
    delta = 1e-5
    logical_batch_size = 2
    adjacency = "add_remove"

  calibrations = _calibrations(Config(), 2)
  assert set(calibrations) == set(TARGET_EPSILONS)
  assert calibrations[3].clip_norm == 1.0
  assert calibrations[3].delta == 1e-5
  assert calibrations[8].iid_noise_std < calibrations[3].iid_noise_std


def test_npz_csv_schema_and_numerical_identity(tmp_path):
  class Result:
    epsilons = np.asarray([3, 8], np.int32)
    steps = np.asarray(REQUIRED_STEPS, np.int32)
    layers = TARGET_LAYER_NAMES
    clean_singular_values = np.asarray(
        [[[[4., 2.], [3., 1.], [2., .5]]]] * 2
    )
    dp_singular_values = np.asarray([
        [[[5., 1.], [4., .8], [2.5, .4]]],
        [[[4.5, 1.2], [3.5, .7], [2.2, .3]]],
    ])
    # Shape the compact fixture to [2 epsilon, 3 steps, 3 layers, 2 index].
    clean_singular_values = np.broadcast_to(
        np.asarray([[[4., 2.], [3., 1.], [2., .5]]]), (2, 3, 3, 2)
    ).copy()
    dp_singular_values = np.broadcast_to(
        np.asarray([[[5., 1.], [4., .8], [2.5, .4]]]), (2, 3, 3, 2)
    ).copy()
    dp_singular_values[1] = np.asarray([[[4.5, 1.2], [3.5, .7], [2.2, .3]]] * 3)

  spectra_path = save_spectra(tmp_path / "spectra.npz", result=Result())
  loaded = load_spectra(spectra_path)
  assert set(loaded) == {
      "epsilons", "steps", "layers", "clean_singular_values", "dp_singular_values"
  }
  csv_path = save_spectra_csv(spectra_path, tmp_path / "spectra.csv")
  with csv_path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.reader(stream))
  assert rows[0] == ["epsilon", "step", "layer", "index", "clean", "dp"]
  assert len(rows) - 1 == loaded["clean_singular_values"].size
  for row in rows[1:]:
    epsilon_index = TARGET_EPSILONS.index(int(row[0]))
    step_index = REQUIRED_STEPS.index(int(row[1]))
    layer_index = TARGET_LAYER_NAMES.index(row[2])
    singular_index = int(row[3]) - 1
    assert float(row[4]) == float(loaded["clean_singular_values"][epsilon_index, step_index, layer_index, singular_index])
    assert float(row[5]) == float(loaded["dp_singular_values"][epsilon_index, step_index, layer_index, singular_index])


def test_two_3x3_plots_are_generated_from_npz(tmp_path):
  result = run_smoke(tmp_path)
  for epsilon in TARGET_EPSILONS:
    plot = plot_singular_spectra(
        tmp_path / "spectra.npz",
        tmp_path / f"singular_spectra_eps{epsilon}.png",
        epsilon=epsilon,
    )
    assert plot.is_file() and plot.stat().st_size > 0
  assert result.clean_singular_values.shape[:3] == (2, 3, 3)


def test_smoke_writes_all_required_artifacts(tmp_path):
  result = run_smoke(tmp_path)
  assert result.epsilons.tolist() == [3, 8]
  assert result.steps.tolist() == [1, 2, 3]
  for name in (
      "spectra.npz",
      "spectra.csv",
      "singular_spectra_eps3.png",
      "singular_spectra_eps8.png",
  ):
    assert (tmp_path / name).is_file()

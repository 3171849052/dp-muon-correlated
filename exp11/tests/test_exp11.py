"""Tests for Exp11 pairing, pre-Q capture, artifacts, and smoke wiring."""

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
from dp_muon.training import (
    init_nonamplified_dpmuon_state,
    make_nonamplified_dpmuon_train_step,
)
import dp_muon.training.nonamplified_dpmuon as dpmuon
from exp11.plotting import (
    load_spectra,
    plot_singular_spectra,
    save_spectra,
    save_spectra_csv,
)
from exp11.run import (
    DPMuonSettings,
    PARAMETER_PATH,
    PARAMETER_NAME,
    run_paired_trajectories,
    run_smoke,
)


def _params():
  def dense():
    return {"kernel": jnp.eye(2, dtype=jnp.float32), "bias": jnp.zeros((2,), jnp.float32)}
  return {
      "blocks": ({
          "attention": {name: dense() for name in ("query", "key", "value", "out")},
          "mlp": {name: dense() for name in ("dense0", "dense1")},
      },),
      "head": dense(),
  }


def _loss(params, batch):
  return (
      jnp.sum(params["blocks"][0]["attention"]["query"]["kernel"])
      + jnp.sum(params["head"]["kernel"])
  ) * batch["scale"][0]


def _calibration(noise_std=0.25):
  calibration = calibrate_nonamplified_iid(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=2.0,
      adjacency="add_remove",
      max_participations=2,
  )
  return replace(calibration, iid_noise_std=noise_std)


def _settings():
  return DPMuonSettings(
      muon_learning_rate=0.01,
      muon_weight_decay=0.0,
      momentum=0.8,
      ns_steps=2,
      consistent_rms=0.2,
      adamw_learning_rate=0.01,
      adamw_beta1=0.9,
      adamw_beta2=0.99,
      adamw_eps=1e-6,
      adamw_weight_decay=0.0,
      microbatch_size=None,
      use_bf16_ns=False,
  )


def _batch(value):
  return {"scale": jnp.asarray([value, -value], dtype=jnp.float32)}


def test_clean_skips_noise_but_still_runs_the_clipped_query(monkeypatch):
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
  step, optimizer = make_nonamplified_dpmuon_train_step(
      _loss, _calibration(), muon_learning_rate=.01, muon_weight_decay=0.0,
      momentum=.8, ns_steps=2, consistent_rms=.2, adamw_learning_rate=.01,
      adamw_beta1=.9, adamw_beta2=.99, adamw_eps=1e-6, adamw_weight_decay=0.0,
      add_noise=False,
  )
  state = init_nonamplified_dpmuon_state(_params(), jax.random.key(0), optimizer)
  step(state, _batch(10.0))
  assert len(query_outputs) == 1
  norm = jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(query_outputs[0])))
  assert float(norm) <= 1.0 / 2.0 + 1e-5


def test_dp_path_uses_one_iid_gaussian_noise_sample(monkeypatch):
  calls = []
  real_sampler = dpmuon.sample_iid_gaussian_noise

  def fixed_noise(key, template, noise_std):
    noise, next_key = real_sampler(key, template, noise_std)
    calls.append((template, noise, noise_std, key, next_key))
    return noise, next_key

  monkeypatch.setattr(dpmuon, "_sample_iid_gaussian_noise", fixed_noise)
  step, optimizer = make_nonamplified_dpmuon_train_step(
      _loss, _calibration(), muon_learning_rate=.01, muon_weight_decay=0.0,
      momentum=.8, ns_steps=2, consistent_rms=.2, adamw_learning_rate=.01,
      adamw_beta1=.9, adamw_beta2=.99, adamw_eps=1e-6, adamw_weight_decay=0.0,
      add_noise=True,
  )
  state = init_nonamplified_dpmuon_state(_params(), jax.random.key(1), optimizer)
  step(state, _batch(1.0))
  assert len(calls) == 1
  template, noise, noise_std, key, next_key = calls[0]
  assert float(noise_std) == .25
  assert jax.tree_util.tree_structure(template) == jax.tree_util.tree_structure(state.params)
  assert jax.tree_util.tree_structure(noise) == jax.tree_util.tree_structure(template)
  assert not np.array_equal(jax.random.key_data(key), jax.random.key_data(next_key))


def test_pre_q_hook_is_after_nesterov_and_before_q():
  beta = .5
  first = jnp.asarray([[2.0, 0.0], [0.0, 1.0]], jnp.float32)
  second = jnp.asarray([[0.0, 3.0], [4.0, 0.0]], jnp.float32)
  params = {"w": first}
  transform = muon_transform(
      learning_rate=.1, weight_decay=0.0, momentum=beta, ns_steps=2,
      consistent_rms=.2, use_bf16_ns=False,
      pre_q_hook=make_pre_q_svd_hook(("w",)),
  )
  state = transform.init(params)
  _, state = transform.update({"w": first}, state, params)
  _, state = transform.update({"w": second}, state, params)
  old_momentum = (1.0 - beta) * first
  new_momentum = beta * old_momentum + (1.0 - beta) * second
  pre_q = (1.0 - beta) * second + beta * new_momentum
  expected = np.linalg.svd(np.asarray(pre_q), compute_uv=False)
  actual = np.asarray(extract_pre_q_singular_values(state))
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
  # The returned update has already passed Q and therefore is not the captured
  # pre-Q matrix for this deliberately non-orthogonal input.
  final_update, _ = transform.update({"w": second}, state, params)
  assert not np.allclose(np.asarray(final_update["w"]), np.asarray(pre_q))


def test_svd_is_descending_and_has_matrix_rank_dimension():
  params = {"w": jnp.zeros((3, 2), jnp.float32)}
  hook = make_pre_q_svd_hook(("w",))
  state = hook.init(params)
  _, state = hook.update({"w": jnp.asarray([[0., 0.], [4., 0.], [0., 1.]])}, state)
  values = np.asarray(state.singular_values)
  np.testing.assert_allclose(values, [4.0, 1.0])
  assert values.shape == (min(params["w"].shape),)
  assert np.all(np.diff(values) <= 0)


def test_pairing_uses_same_initialization_and_only_requested_steps():
  batches = [_batch(1.0), _batch(2.0), _batch(3.0)]
  result = run_paired_trajectories(
      initial_params=_params(), batches=batches, horizon=3,
      calibration=_calibration(), loss_fn=_loss, settings=_settings(),
      seed=4, steps=(1, 3),
  )
  np.testing.assert_array_equal(result.steps, [1, 3])
  assert result.clean_singular_values.shape == (2, 2)
  assert result.dp_singular_values.shape == (2, 2)
  assert not np.array_equal(result.clean_singular_values[1], result.dp_singular_values[1])


def test_npz_schema_and_shared_y_plot(tmp_path):
  spectra = save_spectra(
      tmp_path / "spectra.npz",
      steps=np.asarray([32, 244, 480]),
      parameter_name=PARAMETER_NAME,
      clean_singular_values=np.asarray([[4., 2.], [3., 1.5], [2., 1.]]),
      dp_singular_values=np.asarray([[5., 1.], [3.5, .8], [2.5, .5]]),
  )
  loaded = load_spectra(spectra)
  assert set(loaded) == {"steps", "parameter_name", "clean_singular_values", "dp_singular_values"}
  assert str(loaded["parameter_name"].item()) == PARAMETER_NAME
  csv_path = save_spectra_csv(spectra, tmp_path / "spectra.csv")
  with csv_path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.reader(stream))
  assert rows[0] == ["step", "index", "clean", "dp"]
  assert len(rows) - 1 == loaded["clean_singular_values"].size
  cursor = 1
  for step, clean_spectrum, dp_spectrum in zip(
      loaded["steps"], loaded["clean_singular_values"],
      loaded["dp_singular_values"], strict=True
  ):
    for index, (clean_value, dp_value) in enumerate(
        zip(clean_spectrum, dp_spectrum, strict=True), start=1
    ):
      row = rows[cursor]
      assert int(row[0]) == int(step)
      assert int(row[1]) == index
      assert float(row[2]) == float(clean_value)
      assert float(row[3]) == float(dp_value)
      cursor += 1
  plot = plot_singular_spectra(spectra, tmp_path / "singular_spectra.png")
  assert plot.is_file() and plot.stat().st_size > 0


def test_smoke_runs_both_paths_and_writes_artifacts(tmp_path):
  result = run_smoke(tmp_path)
  assert result.steps.tolist() == [1, 2, 3]
  assert (tmp_path / "spectra.npz").is_file()
  assert (tmp_path / "spectra.csv").is_file()
  assert (tmp_path / "singular_spectra.png").is_file()
  with (tmp_path / "spectra.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.reader(stream))
  assert rows[0] == ["step", "index", "clean", "dp"]
  assert len(rows) - 1 == result.clean_singular_values.size

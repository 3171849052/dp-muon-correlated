"""Lightweight checks for Public-(V) + Frozen AdamW + BandInvMF."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from opacus.accountants.analysis import gdp

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.data import (
    DEFAULT_CIFAR100_PUBLIC_CLASSES,
    filter_cifar100_public_classes,
    load_public_private_cifar,
    split_cifar10_public_private,
)
from dp_muon.optim import PublicVAdamW, public_v_adamw_segment_workload_matrix
from dp_muon.privacy import PrivacyCalibration
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
from dp_muon.training.cifar10_experiment import derive_fixed_cycle_participation
from dp_muon.training.cifar10_public_v_bandinv_experiment import (
    load_cifar10_public_v_bandinv_config,
)
from dp_muon.training.nonamplified_public_v_bandinv import (
    SegmentedBandInvPrivacyAccountant,
    begin_public_v_segment,
    init_public_v_bandinv_adamw_state,
    make_public_v_bandinv_adamw_train_step,
)
from dp_muon.training.public_v import PublicVEstimator


def _images(count: int, offset: int = 0) -> np.ndarray:
  values = np.arange(count * 32 * 32 * 3, dtype=np.uint32) + offset
  return values.astype(np.uint8).reshape(count, 32, 32, 3)


def _write_cifar10(root: Path, count_per_batch: int = 4) -> None:
  directory = root / "cifar-10-batches-py"
  directory.mkdir()
  for batch_index in range(1, 6):
    images = _images(count_per_batch, batch_index)
    values = {
        b"data": images.transpose(0, 3, 1, 2).reshape(count_per_batch, -1),
        b"labels": list(np.arange(count_per_batch) % 10),
    }
    with (directory / f"data_batch_{batch_index}").open("wb") as target:
      pickle.dump(values, target)
  images = _images(2)
  with (directory / "test_batch").open("wb") as target:
    pickle.dump(
        {
            b"data": images.transpose(0, 3, 1, 2).reshape(2, -1),
            b"labels": [0, 1],
        },
        target,
    )


def _write_cifar100(root: Path) -> None:
  directory = root / "cifar-100-python"
  directory.mkdir()
  labels = np.repeat(np.arange(100), 2)
  images = _images(len(labels))
  with (directory / "train").open("wb") as target:
    pickle.dump(
        {
            b"data": images.transpose(0, 3, 1, 2).reshape(len(labels), -1),
            b"fine_labels": labels.tolist(),
        },
        target,
    )


def test_both_public_sources_construct_and_private_contract_uses_private_size(tmp_path):
  _write_cifar10(tmp_path)
  _write_cifar100(tmp_path)
  split = load_public_private_cifar(
      tmp_path,
      public_source="cifar10_split",
      cifar10_public_size=5,
      public_split_seed=17,
      download=False,
  )
  assert len(split.private_images) == 15
  assert len(split.public_images) == 5
  assert not np.intersect1d(
      split.private_source_indices, split.public_source_indices
  ).size
  np.testing.assert_array_equal(
      np.sort(np.concatenate((split.private_source_indices, split.public_source_indices))),
      np.arange(20),
  )
  split_contract = derive_fixed_cycle_participation(15, 2, 5)
  assert (split_contract.horizon, split_contract.min_sep) == (6, 3)

  external = load_public_private_cifar(
      tmp_path,
      public_source="cifar100_10class",
      cifar10_public_size=5,
      public_split_seed=17,
      download=False,
  )
  assert len(external.private_images) == 20
  assert len(external.public_images) == 20
  np.testing.assert_array_equal(np.unique(external.public_labels), np.arange(10))
  external_contract = derive_fixed_cycle_participation(20, 2, 5)
  assert (external_contract.horizon, external_contract.min_sep) == (8, 4)


def test_cifar100_filter_remaps_configured_order_to_zero_through_nine():
  classes = DEFAULT_CIFAR100_PUBLIC_CLASSES[::-1]
  labels = np.asarray([classes[3], 99, classes[0], classes[9]], np.int32)
  images = _images(len(labels))
  filtered_images, remapped, indices = filter_cifar100_public_classes(
      images, labels, classes
  )
  np.testing.assert_array_equal(indices, [0, 2, 3])
  np.testing.assert_array_equal(remapped, [3, 0, 9])
  np.testing.assert_array_equal(filtered_images, images[indices])


def test_cifar10_split_is_fixed_for_seed_and_disjoint():
  images, labels = _images(13), np.arange(13, dtype=np.int32) % 10
  first = split_cifar10_public_private(
      images, labels, public_size=4, split_seed=9
  )
  second = split_cifar10_public_private(
      images, labels, public_size=4, split_seed=9
  )
  np.testing.assert_array_equal(first.public_source_indices, second.public_source_indices)
  assert not np.intersect1d(
      first.private_source_indices, first.public_source_indices
  ).size


def test_yaml_switches_public_source_without_training_code_changes(tmp_path):
  source = Path(__file__).parents[1] / "config/cifar10_public_v_bandinv.yaml"
  config = load_cifar10_public_v_bandinv_config(source)
  assert config.public_source == "cifar10_split"
  assert config.public_v_beta2 == 0.999
  assert config.public_v_eps == 1e-8
  assert config.public_v_batches_per_segment == 4
  assert config.segment_length == 64
  changed = tmp_path / "external.yaml"
  changed.write_text(
      source.read_text(encoding="utf-8").replace(
          "public_source: cifar10_split", "public_source: cifar100_10class"
      ),
      encoding="utf-8",
  )
  assert (
      load_cifar10_public_v_bandinv_config(changed).public_source
      == "cifar100_10class"
  )


def _params():
  return {
      "w": jnp.asarray([0.5, -0.25], jnp.float32),
      "bias": jnp.asarray(0.1, jnp.float32),
  }


def _public_loss(params, batch):
  prediction = batch["x"] @ params["w"] + params["bias"]
  return jnp.mean(jnp.square(prediction - batch["y"]))


def _private_loss(params, batch):
  prediction = batch["x"][0] @ params["w"] + params["bias"]
  return jnp.square(prediction - batch["y"][0])


def _batch(scale: float = 1.0):
  return {
      "x": jnp.asarray([[1.0, 2.0], [-1.0, 0.5]], jnp.float32) * scale,
      "y": jnp.asarray([0.25, -0.5], jnp.float32),
  }


def _calibration():
  return PrivacyCalibration(
      epsilon=1.0,
      delta=1e-5,
      adjacency="add_remove",
      clip_norm=10.0,
      normalize_by=2.0,
      query_sensitivity=5.0,
      matrix_sensitivity=1.0,
      total_sensitivity=5.0,
      mu=1.0,
      noise_multiplier=1.0,
      iid_noise_std=0.0,
      max_participations=2,
  )


def _fake_fit(horizon, bandwidth, min_sep, **kwargs):
  assert kwargs["workload_matrix"].shape == (horizon, horizon)
  return BandInvMFStrategy(
      horizon=horizon,
      bandwidth=bandwidth,
      min_sep=min_sep,
      max_participations=kwargs["max_participations"],
      workload_coef=None,
      noising_coef=jnp.ones((bandwidth,), jnp.float32),
      strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=jnp.asarray(1.0, jnp.float32),
      objective=jnp.asarray(0.0, jnp.float32),
      workload_matrix=jnp.asarray(kwargs["workload_matrix"]),
  )


def _assert_tree_equal(left, right):
  for actual, expected in zip(
      jax.tree_util.tree_leaves(left),
      jax.tree_util.tree_leaves(right),
      strict=True,
  ):
    if jnp.issubdtype(jnp.asarray(actual).dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(
          jax.random.key_data(actual), jax.random.key_data(expected)
      )
    else:
      np.testing.assert_array_equal(actual, expected)


def test_public_v_shape_device_dtype_and_counter_are_private_step_independent():
  params = _params()
  estimator = PublicVEstimator(_public_loss, beta2=0.8, eps=1e-6)
  public_state = jax.jit(estimator.update)(
      estimator.init(params), params, _batch()
  )
  assert int(public_state.t_v) == 1
  for parameter, value in zip(
      jax.tree_util.tree_leaves(params),
      jax.tree_util.tree_leaves(public_state.v),
      strict=True,
  ):
    assert value.shape == parameter.shape
    assert value.dtype == parameter.dtype
    assert value.device == parameter.device
  optimizer = PublicVAdamW(learning_rate=0.05, beta1=0.9, eps=1e-6)
  optimizer_state = optimizer.set_public_v(
      optimizer.init(params), estimator.bias_corrected_v(public_state), params
  )
  _, optimizer_state = jax.jit(optimizer.update)(
      jax.tree_util.tree_map(jnp.ones_like, params), optimizer_state, params
  )
  assert int(optimizer_state.count) == 1
  assert int(public_state.t_v) == 1


def test_frozen_v_segment_transition_and_synthetic_private_chain(tmp_path):
  params = _params()
  estimator = PublicVEstimator(_public_loss, beta2=0.8, eps=1e-6)
  optimizer = PublicVAdamW(
      learning_rate=0.05, beta1=0.7, eps=1e-6, weight_decay=0.01
  )
  state = init_public_v_bandinv_adamw_state(
      params,
      optimizer=optimizer,
      estimator=estimator,
      noise_root_key=jax.random.key(3),
      bandwidth=2,
  )
  state, first_info = begin_public_v_segment(
      state,
      [_batch()],
      estimator=estimator,
      optimizer=optimizer,
      segment_index=0,
      segment_length=2,
      global_min_sep=3,
      bandwidth=2,
      num_segments=2,
      global_noise_multiplier=0.0,
      query_sensitivity=1.0,
      learning_rates=0.05,
      reduction="mean",
      max_optimizer_steps=1,
      fit_strategy=_fake_fit,
  )
  frozen = state.optimizer_state.public_v_hat
  assert int(state.public_v_state.t_v) == 1
  assert first_info.workload_matrix.shape == (2, 2)
  step = jax.jit(
      make_public_v_bandinv_adamw_train_step(
          _private_loss, _calibration(), optimizer
      )
  )
  state = step(state, _batch())
  state = step(state, _batch())
  _assert_tree_equal(state.optimizer_state.public_v_hat, frozen)
  assert int(state.public_v_state.t_v) == 1
  assert int(state.optimizer_state.count) == 2

  params_before = state.params
  momentum_before = state.optimizer_state.mu
  state, second_info = begin_public_v_segment(
      state,
      [_batch(2.0)],
      estimator=estimator,
      optimizer=optimizer,
      segment_index=1,
      segment_length=1,
      global_min_sep=3,
      bandwidth=2,
      num_segments=2,
      global_noise_multiplier=0.0,
      query_sensitivity=1.0,
      learning_rates=0.05,
      reduction="mean",
      max_optimizer_steps=1,
      fit_strategy=_fake_fit,
  )
  _assert_tree_equal(state.params, params_before)
  _assert_tree_equal(state.optimizer_state.mu, momentum_before)
  assert int(state.optimizer_state.count) == 2
  assert int(state.public_v_state.t_v) == 2
  assert int(state.noise_state.step) == 0
  assert second_info.workload_matrix.shape == (1, 1)
  assert any(
      not np.array_equal(left, right)
      for left, right in zip(
          jax.tree_util.tree_leaves(frozen),
          jax.tree_util.tree_leaves(state.optimizer_state.public_v_hat),
          strict=True,
      )
  )
  final = step(state, _batch())
  assert int(final.step) == 3
  assert int(final.optimizer_state.count) == 3
  assert int(final.public_v_state.t_v) == 2

  checkpoint = tmp_path / "state.pkl"
  save_checkpoint(
      checkpoint,
      state=final,
      current_step=3,
      experiment_config={"algorithm": "public-v-test"},
      artifact_identifiers={"algorithm": "dp-adamw-public-v-bandinv"},
  )
  restored = load_checkpoint(checkpoint)["state"]
  _assert_tree_equal(restored, final)

  accountant = SegmentedBandInvPrivacyAccountant(
      num_segments=2, global_mu=0.5, delta=1e-5
  )
  accountant.set_current_state(final)
  np.testing.assert_allclose(
      accountant.epsilon_spent(3),
      gdp.eps_from_mu(mu=0.5, delta=1e-5),
  )


def test_segment_workload_matches_explicit_frozen_v_adamw_recurrence():
  length, beta, start = 3, 0.8, 4
  rates = np.asarray([0.1, 0.05, 0.02])
  weight_decay = 0.2
  actual = np.asarray(
      public_v_adamw_segment_workload_matrix(
          length,
          beta,
          rates,
          weight_decay,
          first_moment_start_step=start,
      )
  )
  expected = np.zeros((length, length))
  for query in range(length):
    moment = 0.0
    parameter = 0.0
    for step_index in range(length):
      gradient = 1.0 if step_index == query else 0.0
      moment = beta * moment + (1.0 - beta) * gradient
      corrected = moment / (1.0 - beta ** (start + step_index + 1))
      parameter = (
          (1.0 - rates[step_index] * weight_decay) * parameter
          - rates[step_index] * corrected
      )
      expected[step_index, query] = -parameter
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
  np.testing.assert_array_equal(np.triu(actual, k=1), np.zeros((length, length)))


def test_public_v_does_not_change_temporal_workload_passed_to_bandinv():
  params = _params()
  estimator = PublicVEstimator(_public_loss, beta2=0.8, eps=1e-6)
  optimizer = PublicVAdamW(
      learning_rate=0.05, beta1=0.7, eps=1e-6, weight_decay=0.01
  )

  def begin_with_public_batch(public_batch):
    initial = init_public_v_bandinv_adamw_state(
        params,
        optimizer=optimizer,
        estimator=estimator,
        noise_root_key=jax.random.key(11),
        bandwidth=2,
    )
    return begin_public_v_segment(
        initial,
        [public_batch],
        estimator=estimator,
        optimizer=optimizer,
        segment_index=0,
        segment_length=2,
        global_min_sep=3,
        bandwidth=2,
        num_segments=1,
        global_noise_multiplier=0.0,
        query_sensitivity=1.0,
        learning_rates=np.asarray([0.05, 0.025]),
        reduction="mean",
        max_optimizer_steps=1,
        fit_strategy=_fake_fit,
    )

  first_state, first_info = begin_with_public_batch(_batch())
  second_state, second_info = begin_with_public_batch(_batch(2.0))

  assert not np.isclose(
      first_info.preconditioner_rms, second_info.preconditioner_rms
  )
  assert any(
      not np.array_equal(left, right)
      for left, right in zip(
          jax.tree_util.tree_leaves(first_state.optimizer_state.public_v_hat),
          jax.tree_util.tree_leaves(second_state.optimizer_state.public_v_hat),
          strict=True,
      )
  )
  np.testing.assert_array_equal(
      first_info.workload_matrix, second_info.workload_matrix
  )
  np.testing.assert_array_equal(
      first_info.strategy.workload_matrix, second_info.strategy.workload_matrix
  )

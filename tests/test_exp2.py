"""Exp2 orchestration and privacy-contract tests."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import yaml

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import (
    load_cifar10_bandinv_dpadamw_config,
)
from exp2.common import derive_contract
from exp2.full_training import calibration_metadata
from exp2.strategies import ADAM_M_AWARE, DECAYED_PREFIX, StrategySpec, workload_for


def test_full_horizon_contract_is_derived_from_real_config_values():
  config = load_cifar10_bandinv_dpadamw_config("config/cifar10_bandinv_dpadamw_naive.yaml")
  contract = derive_contract(config, num_examples=50_000)
  assert (contract.horizon, contract.min_sep, contract.max_participations) == (488, 97, 5)
  assert config.learning_rate == 0.0005


def test_exp2_config_does_not_duplicate_participation_contract():
  document = yaml.safe_load(open("exp2/config.yaml", encoding="utf-8"))
  assert "min_sep" not in document["strategy"]
  assert "max_participations" not in document["strategy"]


def test_strategy_workload_representations_are_distinct_and_required():
  naive = StrategySpec(DECAYED_PREFIX, 8, 4, 3, 5, 0.0005, 0.9, 0.01)
  aware = StrategySpec(ADAM_M_AWARE, 8, 4, 3, 5, 0.0005, 0.9, 0.01)
  assert workload_for(naive)["workload_coef"].shape == (8,)
  matrix = np.asarray(workload_for(aware)["workload_matrix"])
  assert matrix.shape == (8, 8)
  assert np.allclose(np.triu(matrix, 1), 0.0)


def test_training_calibration_uses_sensitivity_not_relative_noise_target():
  config = SimpleNamespace(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, batch_size=512, adjacency="add_remove"
  )
  def strategy(sensitivity):
    return BandInvMFStrategy(
        horizon=3, bandwidth=1, min_sep=2, max_participations=2,
        workload_coef=jnp.ones(3), workload_matrix=None,
        noising_coef=jnp.ones(1), strategy_coef=jnp.ones(3),
        sensitivity_squared=jnp.asarray(sensitivity), objective=jnp.asarray(2.0),
    )
  naive = calibration_metadata(config, strategy(1.0))
  aware = calibration_metadata(config, strategy(4.0))
  assert (naive["epsilon"], naive["delta"]) == (aware["epsilon"], aware["delta"])
  assert aware["calibrated_noise_stddev"] == 2 * naive["calibrated_noise_stddev"]
  assert aware["calibrated_noise_multiplier"] == naive["calibrated_noise_multiplier"]


def test_training_calibration_keeps_common_budget_and_strategy_specific_stddev():
  config = SimpleNamespace(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, batch_size=512, adjacency="add_remove"
  )
  def strategy(sensitivity):
    return BandInvMFStrategy(
        horizon=3, bandwidth=1, min_sep=2, max_participations=2,
        workload_coef=jnp.ones(3), workload_matrix=None,
        noising_coef=jnp.ones(1), strategy_coef=jnp.ones(3),
        sensitivity_squared=jnp.asarray(sensitivity), objective=jnp.asarray(2.0),
    )
  naive = calibration_metadata(config, strategy(1.0))
  aware = calibration_metadata(config, strategy(9.0))
  assert (naive["epsilon"], naive["delta"]) == (aware["epsilon"], aware["delta"])
  assert aware["calibrated_noise_stddev"] == 3 * naive["calibrated_noise_stddev"]
  assert aware["calibrated_noise_multiplier"] == naive["calibrated_noise_multiplier"]

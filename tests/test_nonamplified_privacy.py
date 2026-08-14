import math

import numpy as np
import pytest
from opacus.accountants.analysis import gdp

from dp_muon.bandinvmf import fit_bandinv_strategy
from dp_muon.privacy import calibrate_nonamplified_bandinv
from scripts.calibrate_nonamplified import load_sensitivity_squared


def _calibrate(**overrides):
  parameters = dict(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=1.5,
      normalize_by=6.0,
      adjacency="add_remove",
      sensitivity_squared=9.0,
  )
  parameters.update(overrides)
  return calibrate_nonamplified_bandinv(**parameters)


def test_gdp_round_trip_uses_opacus_conversion():
  result = _calibrate()
  recovered_epsilon = gdp.eps_from_mu(mu=result.mu, delta=result.delta)
  assert recovered_epsilon == pytest.approx(result.epsilon, rel=1e-10)


def test_noise_scales_only_with_sensitivity_inputs():
  baseline = _calibrate()
  cases = (
      (_calibrate(clip_norm=3.0), 2.0),
      (_calibrate(normalize_by=12.0), 0.5),
      (_calibrate(sensitivity_squared=36.0), 2.0),
  )
  for changed, expected_tau_ratio in cases:
    assert changed.iid_noise_std == pytest.approx(
        baseline.iid_noise_std * expected_tau_ratio
    )
    assert changed.mu == pytest.approx(baseline.mu)
    assert changed.noise_multiplier == pytest.approx(baseline.noise_multiplier)


def test_replace_one_doubles_add_remove_noise():
  add_remove = _calibrate(adjacency="add_remove")
  replace_one = _calibrate(adjacency="replace_one")
  assert replace_one.iid_noise_std == pytest.approx(2.0 * add_remove.iid_noise_std)


def test_strategy_artifact_integration(tmp_path):
  strategy = fit_bandinv_strategy(8, 3, 1, max_optimizer_steps=3)
  artifact_path = tmp_path / "tiny_strategy.npz"
  np.savez(artifact_path, sensitivity_squared=np.asarray(strategy.sensitivity_squared))
  result = _calibrate(sensitivity_squared=load_sensitivity_squared(artifact_path))
  assert result.matrix_sensitivity == pytest.approx(
      math.sqrt(float(strategy.sensitivity_squared))
  )
  assert result.iid_noise_std == pytest.approx(
      result.noise_multiplier
      * result.query_sensitivity
      * result.matrix_sensitivity
  )


@pytest.mark.parametrize(
    "overrides",
    (
        {"epsilon": 0},
        {"epsilon": float("nan")},
        {"delta": 0},
        {"delta": 1},
        {"delta": float("inf")},
        {"clip_norm": 0},
        {"normalize_by": 0},
        {"sensitivity_squared": 0},
        {"sensitivity_squared": float("nan")},
        {"adjacency": "unknown"},
    ),
)
def test_invalid_parameters_fail_fast(overrides):
  with pytest.raises(ValueError):
    _calibrate(**overrides)

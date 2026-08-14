import importlib

import jax.numpy as jnp
import numpy as np

from dp_muon.bandinvmf import fit_bandinv_strategy
from scripts.fit_bandinvmf import default_artifact_path


def test_required_jax_privacy_api_is_importable():
  toeplitz = importlib.import_module("jax_privacy.matrix_factorization.toeplitz")
  for name in (
      "optimize_banded_inverse_toeplitz",
      "compute_banded_inverse_sensitivity_squared",
      "inverse_coef",
      "per_query_error",
      "banded_inverse_square_root_noising_coefs",
  ):
    assert callable(getattr(toeplitz, name))


def test_tiny_bandinv_strategy_has_valid_inverse_and_sensitivity():
  horizon = 8
  result = fit_bandinv_strategy(
      horizon, bandwidth=3, min_sep=1, max_optimizer_steps=3
  )
  convolution = jnp.convolve(result.noising_coef, result.strategy_coef)[:horizon]
  np.testing.assert_allclose(np.asarray(convolution), np.eye(1, horizon, 0)[0], atol=3e-4)
  sensitivity = float(result.sensitivity_squared)
  assert np.isfinite(sensitivity)
  assert sensitivity > 0
  assert np.isfinite(float(result.objective))


def test_reduction_is_reflected_in_the_objective():
  result = fit_bandinv_strategy(
      8, bandwidth=3, min_sep=1, max_optimizer_steps=3, reduction="last"
  )
  toeplitz = importlib.import_module("jax_privacy.matrix_factorization.toeplitz")
  errors = toeplitz.per_query_error(
      noising_coef=result.noising_coef,
      n=result.horizon,
      workload_coef=result.workload_coef,
  )
  np.testing.assert_allclose(
      float(result.objective), float(errors[-1] * result.sensitivity_squared)
  )


def test_default_artifact_name_identifies_n_p_b_and_k(tmp_path):
  uncapped = default_artifact_path(
      tmp_path, horizon=16, bandwidth=4, min_sep=2, max_participations=None
  )
  capped = default_artifact_path(
      tmp_path, horizon=16, bandwidth=4, min_sep=2, max_participations=3
  )
  assert uncapped.name == "prefix_n16_p4_b2_kmax.npz"
  assert capped.name == "prefix_n16_p4_b2_k3.npz"
  assert uncapped != capped

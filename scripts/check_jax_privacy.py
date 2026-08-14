#!/usr/bin/env python3
"""Report the installed jax_privacy API and run a tiny Toeplitz smoke test."""

from __future__ import annotations

import importlib.metadata
import inspect

import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy.matrix_factorization import toeplitz


def _metadata() -> str:
  for distribution in ("jax-privacy", "jax_privacy"):
    try:
      dist = importlib.metadata.distribution(distribution)
      return f"{dist.metadata['Name']} {dist.version} ({dist.locate_file('')})"
    except importlib.metadata.PackageNotFoundError:
      pass
  return "package metadata unavailable"


def main() -> None:
  print(f"jax: {jax.__version__} ({jax.__file__})")
  print(f"jax_privacy: {getattr(jax_privacy, '__version__', 'no __version__')} ({jax_privacy.__file__})")
  print(f"jax_privacy metadata: {_metadata()}")
  print(f"toeplitz module: {toeplitz.__file__}")
  names = (
      "optimize_banded_inverse_toeplitz",
      "compute_banded_inverse_sensitivity_squared",
      "inverse_coef",
      "per_query_error",
      "banded_inverse_square_root_noising_coefs",
  )
  for name in names:
    function = getattr(toeplitz, name)
    print(f"{name}{inspect.signature(function)}")

  horizon, bandwidth, min_sep = 8, 3, 1
  workload = jnp.ones(horizon)
  initial = toeplitz.banded_inverse_square_root_noising_coefs(bandwidth, workload)
  noising = toeplitz.optimize_banded_inverse_toeplitz(
      horizon, min_sep, bandwidth, workload_coef=workload, max_optimizer_steps=3
  )
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      horizon, noising, min_sep
  )
  strategy = toeplitz.inverse_coef(noising, horizon)
  errors = toeplitz.per_query_error(noising_coef=noising, n=horizon, workload_coef=workload)
  print("smoke test passed")
  print(f"  BISR initialization: {jnp.asarray(initial)}")
  print(f"  optimized noising_coef (C^-1): {jnp.asarray(noising)}")
  print(f"  strategy_coef (C): {jnp.asarray(strategy)}")
  print(f"  sensitivity_squared: {float(sensitivity):.8g}")
  print(f"  mean per-query error: {float(jnp.mean(errors)):.8g}")


if __name__ == "__main__":
  main()

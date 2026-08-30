"""Experiment 7b: paper-form gamma-prime stabilization for DP-AdamBC."""

from exp7b.core import (
    DEFAULT_GAMMA_PRIME_RATIO,
    gamma_prime_from_ratio,
    make_exp7b_train_step,
    paper_bc_preconditioner,
    phi_infinity,
)

__all__ = [
    "DEFAULT_GAMMA_PRIME_RATIO",
    "gamma_prime_from_ratio",
    "make_exp7b_train_step",
    "paper_bc_preconditioner",
    "phi_infinity",
]

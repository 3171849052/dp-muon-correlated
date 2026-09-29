"""Linear moment-state workloads, without optimizer readout factors."""
import jax.numpy as jnp
from jax_privacy.matrix_factorization.streaming_matrix import StreamingMatrix


def ema(beta):
    def next_value(value, state):
        state = beta * state + (1 - beta) * value
        return state, state
    return StreamingMatrix.from_array_implementation(jnp.zeros_like, next_value)

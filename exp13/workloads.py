"""Streaming raw and bias-corrected EMA utility workloads."""
import jax.numpy as jnp
from jax_privacy.matrix_factorization.streaming_matrix import StreamingMatrix


def ema(beta, corrected=False):
    def initialize(value):
        return jnp.zeros_like(value), jnp.array(0)
    def next_value(value, state):
        old, t = state
        raw = beta * old + (1 - beta) * value
        t = t + 1
        result = raw / (1 - beta**t) if corrected else raw
        return result, (raw, t)
    return StreamingMatrix.from_array_implementation(initialize, next_value)

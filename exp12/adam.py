import jax
import jax.numpy as jnp
import optax

def optimizer(c):
    return optax.adam(learning_rate=c.learning_rate, b1=c.beta1, b2=c.beta2, eps=c.adam_eps)

def recurrence(q, beta1=.9, beta2=.999, eps=1e-8, denominator=None):
    """Time-major inputs; returns bias-corrected directions and preconditioners."""
    def step(state, item):
        m, v = state
        t, g = item
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g*g
        mh = m / (1-beta1**t)
        vh = v / (1-beta2**t)
        pre = 1/(jnp.sqrt(vh)+eps) if denominator is None else jnp.ones_like(vh)/denominator
        return (m, v), (mh*pre, pre)
    zero = jnp.zeros_like(q[0])
    return jax.lax.scan(step, (zero, zero), (jnp.arange(1, len(q)+1), q))[1]

"""Adam with separate moment inputs and a linear raw second state."""
import jax
import jax.numpy as jnp
import optax
from jax_privacy.noise_addition import matrix_factorization_privatizer
from exp13.strategies import noising
from exp13.privacy import calibration


def optimizer(c):
    return optax.adam(c.learning_rate, b1=c.beta1, b2=c.beta2, eps=c.adam_eps)


def square_mean(g):
    return jax.tree.map(jnp.square, g)


def init(g):
    zero = jax.tree.map(jnp.zeros_like, g)
    return jnp.array(0), zero, zero


IME_METHODS = ("iid_ime", "bandmf_ime_sep")


def uses_ime(method):
    return method in IME_METHODS


def moments(state, first, second, c, *, method):
    t, m, v = state
    t = t + 1
    m = jax.tree.map(lambda old, x: c.beta1*old+(1-c.beta1)*x, m, first)
    v = jax.tree.map(lambda old, x: c.beta2*old+(1-c.beta2)*x, v, second)
    mh = jax.tree.map(lambda x:x/(1-c.beta1**t), m)
    v_hat_raw = jax.tree.map(lambda x:x/(1-c.beta2**t), v)
    v_hat_use = jax.tree.map(jnp.abs, v_hat_raw) if uses_ime(method) else v_hat_raw
    direction = jax.tree.map(lambda a,b:a/(jnp.sqrt(b)+c.adam_eps), mh, v_hat_use)
    return (t,m,v), (mh,v_hat_raw,v_hat_use,direction)


def channels(c,p,method,root,key):
    cal=calibration(c,p,method)
    return tuple(matrix_factorization_privatizer(noising(root,method,i), stddev=cal[f'sigma{i}'],
                prng_key=jax.random.fold_in(key,i), dtype=jnp.float32) for i in (1,2))


def private_inputs(g, states, privatizers, method):
    one,two=privatizers
    first,s1=one.update(g,states[0])
    if uses_ime(method):
        second,s2=two.update(square_mean(g),states[1])
    else:
        second,s2=square_mean(first),states[1]
    return first,second,(s1,s2)

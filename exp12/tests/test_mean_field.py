import jax
import jax.numpy as jnp
import numpy as np
from exp12.objectives import expected_v,filter_noise,moment_matrix

def test_expected_v_monte_carlo(c):
    T=8; s=jnp.array([1.8,-.6,.12,-.01]); sigma=.04
    z=jax.random.normal(jax.random.key(17),(T,100000))
    q=sigma*filter_noise(s,z)
    empirical=moment_matrix(T,c.beta2)@jnp.mean(q*q,axis=1)
    np.testing.assert_allclose(empirical,expected_v(s,T,sigma,c.beta2),rtol=.015)

import jax.numpy as jnp
import numpy as np
from exp12.objectives import momentum,moment_matrix,exact,mean_field,toeplitz_matrix

def test_momentum_no_bias(c):
    T=10; H=moment_matrix(T,c.beta1,False)
    np.testing.assert_allclose(momentum(c,T),np.tril(np.ones((T,T)))@H,atol=1e-12)
    assert not np.allclose(momentum(c,T),np.tril(np.ones((T,T)))@moment_matrix(T,c.beta1))

def test_fixed_denominator_linear_limits(c):
    T=8; coef=jnp.array([2.,-.7,.1]); sigma=.03; denominator=.4
    A=jnp.cumsum(moment_matrix(T,c.beta1)@(sigma*toeplitz_matrix(coef,T)),axis=0)/denominator
    expected=jnp.sum(A*A)/T
    # sqrt(T) I has empirical covariance I, giving an exact linear expectation.
    z=jnp.sqrt(T)*jnp.eye(T)
    np.testing.assert_allclose(exact(coef,z,sigma,c,denominator),expected,rtol=1e-10)
    np.testing.assert_allclose(mean_field(coef,T,sigma,c,denominator),expected,rtol=1e-10)

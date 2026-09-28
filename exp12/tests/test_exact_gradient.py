import jax
import jax.numpy as jnp
import numpy as np
from exp12.objectives import exact,mean_field
from exp12.strategies import normalize

def test_exact_gradient(c,p):
    z=jax.random.normal(jax.random.key(123),(p.horizon,256))
    x=jnp.array([1.,-.3,.07,-.01])
    for fn in (lambda s:exact(s,z,.02,c),lambda s:mean_field(s,p.horizon,.02,c)):
        f=lambda x:fn(normalize(x,p))
        grad=jax.grad(f)(x); h=1e-5
        numerical=jnp.array([(f(x+jnp.eye(4)[i]*h)-f(x-jnp.eye(4)[i]*h))/(2*h) for i in range(4)])
        np.testing.assert_allclose(grad,numerical,rtol=2e-4,atol=2e-6)

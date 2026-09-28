import jax
import jax.numpy as jnp
import numpy as np
import optax
from exp12.adam import recurrence, optimizer

def test_recurrence_optax(c):
    q=jax.random.normal(jax.random.key(0),(12,7))
    u,_=recurrence(q,c.beta1,c.beta2,c.adam_eps)
    opt=optimizer(c); params=jnp.zeros(7); state=opt.init(params)
    for i,g in enumerate(q):
        update,state=opt.update(g,state,params)
        np.testing.assert_allclose(update,-c.learning_rate*u[i],rtol=1e-10,atol=1e-12)
        params=optax.apply_updates(params,update)

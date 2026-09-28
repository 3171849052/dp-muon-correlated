import jax
import jax.numpy as jnp
import numpy as np
from exp12.replay import sufficient_statistics,metrics

def test_zero_noise(c):
    g=jax.random.normal(jax.random.key(8),(6,11))
    stats=np.asarray(sufficient_statistics(g,jnp.zeros_like(g),c))[None]
    m=metrics(stats,11)
    for key in ('parameter_trajectory_rmse','endpoint_rmse','adam_direction_rmse','preconditioner_rmse','global_energy_ratio'):
        assert m[key]==0
    np.testing.assert_allclose(m['update_cosine_similarity'],1.)
    assert m['sign_agreement']==1

def test_coordinate_chunk_additivity(c):
    g=jax.random.normal(jax.random.key(1),(5,9)); n=.02*g
    full=sufficient_statistics(g,n,c)
    split=sufficient_statistics(g[:,:4],n[:,:4],c)+sufficient_statistics(g[:,4:],n[:,4:],c)
    np.testing.assert_allclose(full,split,atol=1e-12)

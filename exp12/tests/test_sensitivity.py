import jax.numpy as jnp
import numpy as np
from exp12.strategies import normalize,sensitivity_squared,save

def test_shared_normalization(p,tmp_path):
    for raw in ([1.,-.4,.1,0.],[1.,-.6,.03,.02],[1.,-.2,-.1,.05]):
        raw=jnp.array(raw); s=normalize(raw,p)
        np.testing.assert_allclose(sensitivity_squared(s,p),1.,rtol=1e-10)
        path=tmp_path/'strategy.npz'; save(path,raw,p,2.,{})
        with np.load(path) as f:
            np.testing.assert_allclose(f['sensitivity_after'],1.,rtol=1e-10)
            np.testing.assert_allclose(np.convolve(f['noising_coef'],f['strategy_coef'])[:p.horizon],[1,0,0,0],atol=1e-12)

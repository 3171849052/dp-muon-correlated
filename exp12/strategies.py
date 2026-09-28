import json
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import toeplitz

def sensitivity_squared(coef, p):
    return toeplitz.compute_banded_inverse_sensitivity_squared(n=p.horizon,
        noising_coef=coef, min_sep=p.min_sep, max_participations=p.max_participations,
        use_matrix_upper_bound=False)

def normalize(raw, p):
    return jnp.sqrt(sensitivity_squared(raw,p))*raw

def save(path, raw, p, objective, metadata):
    coef = normalize(raw,p)
    np.savez(path, raw_noising_coef=np.asarray(raw), noising_coef=np.asarray(coef),
        strategy_coef=np.asarray(toeplitz.inverse_coef(coef,p.horizon)),
        sensitivity_before=float(jnp.sqrt(sensitivity_squared(raw,p))),
        sensitivity_after=float(jnp.sqrt(sensitivity_squared(coef,p))),
        objective=float(objective), metadata=json.dumps(metadata))

def coefficients(root, method):
    if method == 'dp_adam':
        return jnp.ones(1)
    with np.load(root/'strategies'/f'{method}.npz') as f:
        return jnp.asarray(f['noising_coef'])

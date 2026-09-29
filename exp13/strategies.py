import json
import importlib.metadata
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import banded, streaming_matrix
from exp13.workloads import ema


def fit(beta, c, p):
    if c.bandwidth > p.min_sep:
        raise ValueError('bandwidth must be <= min_sep')
    strategy = banded.optimize(p.horizon, bands=c.bandwidth, A=ema(beta),
                              max_optimizer_steps=c.fit_steps, reduction_fn=jnp.mean)
    sensitivity = banded.minsep_sensitivity_squared(strategy, p.min_sep, p.max_participations)
    assert sensitivity == p.max_participations
    matrix = np.asarray(strategy.materialize())
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=0), 1., atol=2e-6)
    metadata = dict(horizon=p.horizon, bandwidth=c.bandwidth, min_sep=p.min_sep,
                    max_participations=p.max_participations, sensitivity_squared=sensitivity,
                    objective=float(jnp.mean(banded.per_query_error(strategy, A=ema(beta)))),
                    beta=beta, jax_privacy_version=importlib.metadata.version('jax_privacy'),
                    fit_steps=c.fit_steps, workload='linear EMA moment state')
    return strategy, metadata


def save(path, strategy, metadata):
    np.savez(path, params=np.asarray(strategy.params), C=np.asarray(strategy.materialize()),
             **metadata, metadata=json.dumps(metadata))


def load(root, name):
    with np.load(root/'strategies'/f'{name}.npz') as data:
        return banded.ColumnNormalizedBanded(jnp.asarray(data['params'], dtype=jnp.float32))


def noising(root, method, channel):
    if (method == 'bandmf_single_m' and channel == 1) or method in ('bandmf_ime_sep', 'bandmf_ime_sep_abs'):
        name = 'C_v' if channel == 2 else 'C_m'
        return load(root, name).inverse_as_streaming_matrix()
    return streaming_matrix.identity()

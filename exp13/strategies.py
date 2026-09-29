import json
import importlib.metadata
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import banded, streaming_matrix
from exp13.workloads import ema
from exp13.common import config_hash


def fit(beta, c, p, corrected=False):
    if c.bandwidth > p.min_sep:
        raise ValueError('bandwidth must be <= min_sep')
    strategy = banded.optimize(p.horizon, bands=c.bandwidth, A=ema(beta, corrected),
                              max_optimizer_steps=c.fit_steps, reduction_fn=jnp.mean)
    sensitivity = banded.minsep_sensitivity_squared(strategy, p.min_sep, p.max_participations)
    assert sensitivity == p.max_participations
    matrix = np.asarray(strategy.materialize())
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=0), 1., atol=2e-6)
    metadata = dict(horizon=p.horizon, bandwidth=c.bandwidth, min_sep=p.min_sep,
                    max_participations=p.max_participations, sensitivity_squared=sensitivity,
                    objective=float(jnp.mean(banded.per_query_error(strategy, A=ema(beta, corrected)))),
                    beta=beta, corrected=corrected, jax_privacy_version=importlib.metadata.version('jax_privacy'),
                    fit_steps=c.fit_steps, workload='bias-corrected EMA' if corrected else 'raw EMA', config_hash=config_hash(c,p))
    return strategy, metadata


def save(path, strategy, metadata):
    np.savez(path, params=np.asarray(strategy.params), C=np.asarray(strategy.materialize()),
             **metadata, metadata=json.dumps(metadata))


def load(root, name):
    with np.load(root/'strategies'/f'{name}.npz') as data:
        return banded.ColumnNormalizedBanded(jnp.asarray(data['params'], dtype=jnp.float32))


STRATEGIES = ('C_m_raw', 'C_m_bc', 'C_v_raw', 'C_v_bc')


def strategy_names(method):
    return {
        'nonprivate_adam': (None, None),
        'iid_adam': (None, None),
        'iid_ime': (None, None),
        'bandmf_single_m': ('C_m_raw', None),
        'bandmf_ime_sep_raw': ('C_m_raw', 'C_v_raw'),
        'bandmf_ime_sep_vbc': ('C_m_raw', 'C_v_bc'),
        'bandmf_ime_sep_bc': ('C_m_bc', 'C_v_bc'),
    }[method]


def noising(root, method, channel):
    name = strategy_names(method)[channel-1]
    return load(root, name).inverse_as_streaming_matrix() if name else streaming_matrix.identity()


def validate(root, c, p):
    for name in STRATEGIES:
        path = root/'strategies'/f'{name}.npz'
        with np.load(path) as data:
            meta = json.loads(str(data['metadata']))
            expected = dict(config_hash=config_hash(c,p), horizon=p.horizon,
                bandwidth=c.bandwidth, min_sep=p.min_sep, max_participations=p.max_participations,
                sensitivity_squared=p.max_participations, fit_steps=c.fit_steps,
                beta=c.beta1 if '_m_' in name else c.beta2, corrected=name.endswith('_bc'))
            if any(meta.get(k) != v for k,v in expected.items()):
                raise ValueError(f'Stage 1 strategy/config mismatch: {path}')
            matrix = np.asarray(banded.ColumnNormalizedBanded(jnp.asarray(data['params'])).materialize())
            np.testing.assert_allclose(matrix, data['C'])
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=0), 1, atol=2e-6)

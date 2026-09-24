"""Small synthetic closed-loop check using the production training primitive."""
import jax
import jax.numpy as jnp
import numpy as np
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state, make_nonamplified_bandinv_dpadamw_train_step)
from dp_muon.training.cifar10_driver import build_logical_schedule
from .core import ADAM_FIELDS, write_table


def train_smoke(output, config, contract, strategies):
    x = jnp.asarray(np.random.default_rng(3).normal(size=(contract.num_examples, 2)))
    y = x @ jnp.array([.3, -.2])
    def loss(params, batch):
        return jnp.mean((batch['x'] @ params - batch['y'])**2)
    rows = []
    for name, strategy in strategies.items():
        calibration = calibrate_nonamplified_bandinv(epsilon=config.epsilon,
            delta=config.delta, clip_norm=config.clip_norm, normalize_by=float(config.batch_size),
            adjacency=config.adjacency, sensitivity_squared=float(strategy.sensitivity_squared))
        step, optimizer = make_nonamplified_bandinv_dpadamw_train_step(loss, strategy,
            calibration, ParticipationSpec(contract.horizon, contract.min_sep, contract.max_participations),
            **{f: getattr(config,f) for f in ADAM_FIELDS}, microbatch_size=2)
        state = init_nonamplified_bandinv_dpadamw_state(jnp.zeros(2), strategy,
                                                       jax.random.key(0), optimizer)
        schedule = build_logical_schedule(num_examples=contract.num_examples,
            batch_size=contract.batch_size, strategy=strategy, seed=0)
        update = jax.jit(step)
        for index in schedule:
            state = update(state, dict(x=x[index], y=y[index]))
        assert int(state.step) == contract.horizon
        assert np.isfinite(np.asarray(state.params)).all()
        rows.append(dict(strategy=name, smoke=True, steps=int(state.step),
                         synthetic_loss=float(loss(state.params, dict(x=x,y=y)))))
    write_table(output, 'synthetic_training_smoke', rows)

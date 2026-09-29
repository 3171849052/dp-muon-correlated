import json
import jax.numpy as jnp
import numpy as np
import pytest
from exp13.adam import init, moments, uses_ime
from exp13.common import configuration, METHODS, HERE
from exp13.privacy import calibration


def test_ime_methods_always_use_absolute_raw_second_moment():
    c=configuration(True); g=jnp.array([.1,.2,.3])
    state,read=moments(init(g),g,jnp.array([-2.,0.,4.]),c,method='iid_ime')
    assert np.any(np.asarray(read[1])<0)
    np.testing.assert_array_equal(read[2],jnp.abs(read[1]))
    np.testing.assert_allclose(state[2],(1-c.beta2)*jnp.array([-2.,0.,4.]))
    for method in ('iid_ime','bandmf_ime_sep_raw'):
        assert uses_ime(method)
    assert not any(method.endswith('_abs') for method in METHODS)


def test_absolute_readout_never_enters_raw_recurrence():
    c=configuration(True); g=jnp.array([.1,.2])
    state,read=moments(init(g),g,jnp.array([-3.,1.]),c,method='bandmf_ime_sep_raw')
    expected=c.beta2*state[2]+(1-c.beta2)*jnp.array([-2.,0.])
    next_state,_=moments(state,g,jnp.array([-2.,0.]),c,method='bandmf_ime_sep_raw')
    np.testing.assert_allclose(next_state[2],expected,rtol=1e-6)
    assert float(next_state[2][0])<0


def test_non_ime_methods_keep_raw_nonnegative_second_moment_behavior():
    c=configuration(True); g=jnp.array([-.2,.3])
    state,read=moments(init(g),g,g*g,c,method='iid_adam')
    np.testing.assert_array_equal(read[2],read[1])
    assert np.all(np.asarray(state[2])>=0)
    assert not uses_ime('nonprivate_adam')
    assert not uses_ime('iid_adam')
    assert not uses_ime('bandmf_single_m')

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
    for method in ('iid_ime','bandmf_ime_sep'):
        assert uses_ime(method)
    assert not any(method.endswith('_abs') for method in METHODS)


def test_absolute_readout_never_enters_raw_recurrence():
    c=configuration(True); g=jnp.array([.1,.2])
    state,read=moments(init(g),g,jnp.array([-3.,1.]),c,method='bandmf_ime_sep')
    expected=c.beta2*state[2]+(1-c.beta2)*jnp.array([-2.,0.])
    next_state,_=moments(state,g,jnp.array([-2.,0.]),c,method='bandmf_ime_sep')
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


def test_workload_validation_contains_all_channels_and_valid_ratios():
    root=HERE/'results_smoke/replay'
    records=json.loads((root/'workload_validation.json').read_text())
    assert {(r['method'],r['channel']) for r in records}=={
        ('iid_ime','first'),('iid_ime','second'),
        ('bandmf_ime_sep','first'),('bandmf_ime_sep','second')}
    for record in records:
        assert record['theoretical_state_mse']>0
        assert record['empirical_state_mse']>=0
        assert record['empirical_over_theoretical']==record['empirical_state_mse']/record['theoretical_state_mse']
        assert np.isfinite(record['empirical_over_theoretical'])


def test_replay_denominator_names_and_per_step_shape():
    metrics=json.loads((HERE/'results_smoke/replay/metrics.json').read_text())
    for row in metrics:
        assert row['optimizer_second_moment']=='abs(v_hat_raw)'
        for exponent in (8,7,6,5,4):
            assert f'denominator_lt_1e_minus_{exponent}_fraction' in row
        assert 'denominator_lt_1e6_fraction' not in row
        assert row['first_state_mse']>=0 and row['second_state_mse']>=0
    rows=json.loads((HERE/'results_smoke/replay/per_step_diagnostics.json').read_text())
    assert len(rows)==2*4
    for method in ('iid_ime','bandmf_ime_sep'):
        selected=[row for row in rows if row['method']==method]
        assert [row['step'] for row in selected]==[1,2,3,4]
        assert all(row['denominator_min']>0 for row in selected)
        assert all(row['absolute_direction_max']>=row['absolute_direction_p99'] for row in selected)


def test_training_metadata_records_single_ime_readout_and_privacy():
    root=HERE/'results_smoke/training'
    records=[json.loads((root/f'{method}_seed0/metadata.json').read_text())
             for method in ('iid_ime','bandmf_ime_sep')]
    for record in records:
        assert record['optimizer_second_moment']=='abs(v_hat_raw)'
        assert 'second_moment_readout' not in record
        cal=record['privacy_calibration']
        assert cal['mu1']==cal['mu2']
        assert cal['mu1']**2+cal['mu2']**2==pytest.approx(cal['mu']**2)


def test_ime_smoke_losses_do_not_explode():
    rows=json.loads((HERE/'results_smoke/ime_training_diagnostics.json').read_text())
    assert {row['method'] for row in rows}=={'iid_ime','bandmf_ime_sep'}
    assert all(not row['loss_at_least_1e10'] for row in rows)

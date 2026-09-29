from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exp13.adam import init, moments, channels, private_inputs, readout_for_method
from exp13.common import configuration, HERE
from exp13.privacy import calibration


def test_abs_readout_and_linear_recurrence():
    c=configuration(True)
    first=jnp.array([.1,.2,.3])
    raw={mode:init(first) for mode in ('relu','abs')}
    for second in (jnp.array([-2.,0.,3.]),jnp.array([0.,1.,-4.])):
        previous=raw['abs'][2]
        reads={}
        for mode in raw:
            raw[mode],reads[mode]=moments(raw[mode],first,second,c,mode)
        for a,b in zip(jax.tree.leaves(raw['relu']),jax.tree.leaves(raw['abs'])):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(raw['abs'][2],c.beta2*previous+(1-c.beta2)*second)
        relu,absolute=reads['relu'],reads['abs']
        negative=np.asarray(absolute[1])<0
        assert np.all(np.asarray(absolute[2])>=0)
        np.testing.assert_array_equal(absolute[2][negative],-absolute[1][negative])
        np.testing.assert_array_equal(relu[2][negative],0)
        np.testing.assert_array_equal(absolute[2][~negative],relu[2][~negative])
        np.testing.assert_array_equal(absolute[3][~negative],relu[3][~negative])


@pytest.mark.parametrize('base',['iid_ime','bandmf_ime_sep'])
def test_paired_privacy_noise_and_states(base):
    c=configuration(True); p=SimpleNamespace(horizon=4,min_sep=4,max_participations=1)
    assert calibration(c,p,base)==calibration(c,p,base+'_abs')
    methods=(base,base+'_abs'); g=jnp.linspace(-.1,.1,32)
    priv={m:channels(c,p,m,HERE/'results_smoke',jax.random.key(72)) for m in methods}
    noise={m:tuple(x.init(g) for x in priv[m]) for m in methods}
    raw={m:init(g) for m in methods}
    for t in range(4):
        inputs={}
        for m in methods:
            first,second,noise[m]=private_inputs(g*(t+1),noise[m],priv[m],m)
            inputs[m]=(first,second)
            raw[m],_=moments(raw[m],first,second,c,readout_for_method(m))
        for left,right in ((inputs[base],inputs[base+'_abs']),
                           (raw[base],raw[base+'_abs']),
                           (noise[base],noise[base+'_abs'])):
            for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
                if jax.dtypes.issubdtype(a.dtype,jax.dtypes.prng_key):
                    a,b=jax.random.key_data(a),jax.random.key_data(b)
                np.testing.assert_array_equal(a,b)


def test_non_ime_readout_unchanged():
    for method in ('nonprivate_adam','iid_adam','bandmf_single_m','iid_ime','bandmf_ime_sep'):
        assert readout_for_method(method)=='relu'


def test_replay_paired_diagnostics():
    import json
    root=HERE/'results_smoke/replay'
    records={r['method']:r for r in json.loads((root/'metrics.json').read_text())}
    for base in ('iid_ime','bandmf_ime_sep'):
        a,b=records[base],records[base+'_abs']
        for metric in ('negative_fraction','raw_second_moment_mse','first_moment_mse'):
            assert a[metric]==b[metric]
        for r in (a,b):
            assert r['absolute_direction_median']<=r['absolute_direction_p90']<=r['absolute_direction_p99']
            assert 0<=r['denominator_le_eps_1p01_fraction']<=r['denominator_lt_1e6_fraction']<=r['denominator_lt_1e4_fraction']<=1
    draws={(r['method'],r['draw']):r for r in json.loads((root/'draw_metrics.json').read_text())}
    for ratio in json.loads((root/'paired_readout_ratios.json').read_text()):
        a=draws[ratio['pair'],ratio['draw']]
        b=draws[ratio['pair']+'_abs',ratio['draw']]
        assert ratio['direction_rmse_abs_over_relu']==b['adam_direction_rmse']/a['adam_direction_rmse']
        assert ratio['readout_mse_abs_over_relu']==b['readout_second_moment_mse']/a['readout_second_moment_mse']


@pytest.mark.parametrize('base',['iid_ime','bandmf_ime_sep'])
def test_training_readout_pair_metadata(base):
    import json
    root=HERE/'results_smoke/training'
    records=[json.loads((root/f'{method}_seed0/metadata.json').read_text())
             for method in (base,base+'_abs')]
    for field in ('config','contract','privacy_calibration','seed','initial_test_metrics',
                  'schedule_sha256','pretrained_sha256'):
        assert records[0][field]==records[1][field]
    for field in ('first_strategy','second_strategy'):
        assert records[0]['strategy'][field]==records[1]['strategy'][field]
    assert records[0]['second_moment_readout']=='relu'
    assert records[1]['second_moment_readout']=='abs'

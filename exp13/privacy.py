"""Fixed-denominator zero-out/add-remove, non-amplified GDP accounting."""
import math
from dp_muon.privacy.nonamplified import calibrate_gdp_noise_multiplier


def calibration(c, p, method):
    mu = 1 / calibrate_gdp_noise_multiplier(c.epsilon, c.delta)
    a1 = c.clip_norm / c.batch_size
    a2 = (2 * c.batch_size - 1) * c.clip_norm**2 / c.batch_size**2
    dual = method in ('iid_ime', 'bandmf_ime_sep')
    channel_mu = mu / math.sqrt(2) if dual else mu
    private = method != 'nonprivate_adam'
    return dict(mu=mu, mu1=channel_mu if private else 0., mu2=channel_mu if dual else 0.,
                a1=a1, a2=a2, temporal_sensitivity_squared=p.max_participations,
                sigma1=a1*math.sqrt(p.max_participations)/channel_mu if private else 0.,
                sigma2=a2*math.sqrt(p.max_participations)/channel_mu if dual else 0.,
                epsilon=c.epsilon if private else None, delta=c.delta, amplification=False,
                adjacency='fixed-denominator zero-out/add-remove')

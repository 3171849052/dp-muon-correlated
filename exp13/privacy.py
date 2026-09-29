"""Fixed-denominator zero-out/add-remove, non-amplified GDP accounting."""
import math
from dp_muon.privacy.nonamplified import calibrate_gdp_noise_multiplier


def calibration(c, p, method):
    """Calibration for BandMF single-channel and IME mechanisms only."""
    dual = method in ('iid_ime', 'bandmf_ime_sep_raw', 'bandmf_ime_sep_vbc', 'bandmf_ime_sep_bc')
    if not dual and method != 'bandmf_single_m':
        raise ValueError('Use canonical iid_calibration for iid_adam; clean Adam has no calibration')
    mu = 1 / calibrate_gdp_noise_multiplier(c.epsilon, c.delta)
    a1 = c.clip_norm / c.batch_size
    a2 = (2 * c.batch_size - 1) * c.clip_norm**2 / c.batch_size**2
    channel_mu = mu / math.sqrt(2) if dual else mu
    return dict(mu=mu, mu1=channel_mu, mu2=channel_mu if dual else 0.,
                a1=a1, a2=a2, temporal_sensitivity_squared=p.max_participations,
                sigma1=a1*math.sqrt(p.max_participations)/channel_mu,
                sigma2=a2*math.sqrt(p.max_participations)/channel_mu if dual else 0.,
                epsilon=c.epsilon, delta=c.delta, amplification=False,
                adjacency='fixed-denominator zero-out/add-remove')


def iid_calibration(c,p):
    from dp_muon.privacy.nonamplified import calibrate_nonamplified_iid
    return calibrate_nonamplified_iid(epsilon=c.epsilon, delta=c.delta,
        clip_norm=c.clip_norm, normalize_by=float(c.batch_size),
        adjacency=c.adjacency, max_participations=p.max_participations)

"""Privacy-calibration helpers for the non-amplified baseline."""

from .clipped_query import make_clipped_gradient_query
from .iid_noise import sample_iid_gaussian_noise
from .participation import (
    ParticipationCertificate,
    ParticipationSpec,
    certify_participation_schedule,
    make_fixed_cycle_selection,
    participation_spec_from_strategy,
    theoretical_max_participations,
    validate_fixed_cycle_dataset,
    validate_participation_spec_against_strategy,
)
from .nonamplified import (
    PrivacyCalibration,
    calibrate_gdp_noise_multiplier,
    calibrate_nonamplified_bandinv,
    calibrate_nonamplified_iid,
    compute_query_sensitivity,
    epsilon_spent_for_bandinv_prefix,
    epsilon_spent_for_iid_prefix,
)

__all__ = [
    "PrivacyCalibration",
    "ParticipationCertificate",
    "ParticipationSpec",
    "calibrate_gdp_noise_multiplier",
    "calibrate_nonamplified_bandinv",
    "calibrate_nonamplified_iid",
    "compute_query_sensitivity",
    "epsilon_spent_for_bandinv_prefix",
    "epsilon_spent_for_iid_prefix",
    "certify_participation_schedule",
    "make_fixed_cycle_selection",
    "make_clipped_gradient_query",
    "sample_iid_gaussian_noise",
    "participation_spec_from_strategy",
    "theoretical_max_participations",
    "validate_fixed_cycle_dataset",
    "validate_participation_spec_against_strategy",
]

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
from .hybrid import (
    continuous_hybrid_prefix_sensitivity_squared,
    continuous_hybrid_sensitivity_squared,
    epsilon_spent_for_continuous_hybrid_prefix,
)
from .jme import (
    ShadowJMEPrivacyCalibration,
    bandinv_operator_norm_1_to_2_squared,
    calibrate_shadow_jme,
    epsilon_spent_for_shadow_jme_prefix,
    jme_gamma_and_joint_sensitivity,
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
    "continuous_hybrid_prefix_sensitivity_squared",
    "continuous_hybrid_sensitivity_squared",
    "epsilon_spent_for_continuous_hybrid_prefix",
    "ShadowJMEPrivacyCalibration",
    "bandinv_operator_norm_1_to_2_squared",
    "calibrate_shadow_jme",
    "epsilon_spent_for_shadow_jme_prefix",
    "jme_gamma_and_joint_sensitivity",
    "certify_participation_schedule",
    "make_fixed_cycle_selection",
    "make_clipped_gradient_query",
    "sample_iid_gaussian_noise",
    "participation_spec_from_strategy",
    "theoretical_max_participations",
    "validate_fixed_cycle_dataset",
    "validate_participation_spec_against_strategy",
]

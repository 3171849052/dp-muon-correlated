"""Linear optimizer components used by the Muon baseline."""

from .linear_workload import (
    adam_first_moment_workload_matrix,
    decayed_prefix_sum_workload_coef,
    frozen_p_time_workload,
    fixed_lr_nesterov_decayed_trajectory_workload_coef,
    fixed_lr_nesterov_trajectory_workload_coef,
    nesterov_kernel_coef,
    public_v_adamw_segment_workload_matrix,
)
from .frozen_p_adamw import (
    FrozenPAdamW,
    FrozenPAdamWState,
    freeze_optax_adamw,
    p_star_from_optax,
)
from .public_v_adamw import PublicVAdamW, PublicVAdamWState
from .muon_nesterov import (
    MuonNesterovState,
    init_muon_nesterov_state,
    muon_nesterov_step,
)
from .muon import classic_nesterov_momentum, muon_post_nesterov_transform, muon_transform
from .muon_q import STAGES as MUON_Q_STAGES, muon_q, muon_q_stages
from .parameter_groups import (
    ADAMW,
    MUON,
    count_muon_parameters,
    is_muon_parameter_path,
    vit_muon_parameter_labels,
)
from .sgd_momentum import (
    SGDMomentumState,
    init_sgd_momentum_state,
    sgd_momentum_step,
)

__all__ = [
    "MuonNesterovState",
    "adam_first_moment_workload_matrix",
    "SGDMomentumState",
    "decayed_prefix_sum_workload_coef",
    "frozen_p_time_workload",
    "fixed_lr_nesterov_decayed_trajectory_workload_coef",
    "fixed_lr_nesterov_trajectory_workload_coef",
    "init_muon_nesterov_state",
    "muon_nesterov_step",
    "classic_nesterov_momentum",
    "muon_post_nesterov_transform",
    "muon_transform",
    "MUON_Q_STAGES",
    "muon_q",
    "muon_q_stages",
    "ADAMW",
    "MUON",
    "count_muon_parameters",
    "is_muon_parameter_path",
    "vit_muon_parameter_labels",
    "nesterov_kernel_coef",
    "public_v_adamw_segment_workload_matrix",
    "PublicVAdamW",
    "PublicVAdamWState",
    "FrozenPAdamW",
    "FrozenPAdamWState",
    "freeze_optax_adamw",
    "p_star_from_optax",
    "init_sgd_momentum_state",
    "sgd_momentum_step",
]

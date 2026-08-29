"""Checkpointing for complete non-amplified training state."""

from __future__ import annotations

from pathlib import Path
import pickle
import os
from typing import Any

import jax
import numpy as np

from .nonamplified_dpsgd import NonAmplifiedDPSGDState
from .nonamplified_dpmuon import NonAmplifiedDPMuonState
from .nonamplified_dpadamw import NonAmplifiedDPAdamWState
from .nonamplified_bandinv_dpmuon import NonAmplifiedBandInvDPMuonState
from .nonamplified_bandinv_dpadamw import NonAmplifiedBandInvDPAdamWState
from .nonamplified_bandinv_stp_dpadamw import (
    NonAmplifiedBandInvSTPDPAdamWState,
)
from .nonamplified_frozen_p_bandinv_dpadamw import (
    NonAmplifiedFrozenPBandInvDPAdamWState,
)
from .nonamplified_public_v_bandinv import PublicVBandInvAdamWState
from .nonamplified_linear import NonAmplifiedBandInvState
from .nonamplified_segmented_bandinv_dpadamw import (
    SegmentedBandInvDPAdamWState,
)
from .nonamplified_shadow_jme_bandinv_dpadamw import (
    ShadowJMEBandInvDPAdamWState,
)
from .file_locking import atomic_replace, atomic_temporary_path, file_lock


def _concrete_step(value: Any, name: str) -> int:
  array = np.asarray(jax.device_get(value))
  if array.shape != () or not np.issubdtype(array.dtype, np.integer):
    raise ValueError(f"{name} must be a concrete integer scalar")
  return int(array)


def _validate_steps(
    state: (NonAmplifiedBandInvState | NonAmplifiedBandInvDPMuonState |
            NonAmplifiedBandInvDPAdamWState | NonAmplifiedDPSGDState |
            NonAmplifiedDPMuonState | NonAmplifiedDPAdamWState |
            NonAmplifiedBandInvSTPDPAdamWState |
            PublicVBandInvAdamWState | NonAmplifiedFrozenPBandInvDPAdamWState |
            SegmentedBandInvDPAdamWState | ShadowJMEBandInvDPAdamWState),
    current_step: int,
) -> None:
  if not isinstance(current_step, (int, np.integer)) or current_step < 0:
    raise ValueError("current_step must be a non-negative integer")
  if isinstance(state, NonAmplifiedBandInvState):
    nesterov_step = _concrete_step(state.nesterov_state.step, "nesterov_state.step")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    if int(current_step) != nesterov_step or int(current_step) != noise_step:
      raise ValueError("current_step must equal nesterov_state.step and noise_state.step")
  elif isinstance(state, NonAmplifiedBandInvDPMuonState):
    optimizer_step = _concrete_step(state.step, "step")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    if int(current_step) != optimizer_step or int(current_step) != noise_step:
      raise ValueError("current_step must equal state.step and noise_state.step")
  elif isinstance(state, NonAmplifiedBandInvDPAdamWState):
    optimizer_step = _concrete_step(state.step, "step")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    if int(current_step) != optimizer_step or int(current_step) != noise_step:
      raise ValueError("current_step must equal state.step and noise_state.step")
  elif isinstance(state, NonAmplifiedBandInvSTPDPAdamWState):
    optimizer_step = _concrete_step(state.step, "step")
    adam_step = _concrete_step(state.optimizer_state.count, "optimizer_state.count")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    if (
        int(current_step) != optimizer_step
        or int(current_step) != adam_step
        or int(current_step) != noise_step
    ):
      raise ValueError(
          "current_step must equal state.step, optimizer_state.count, and noise_state.step"
      )
  elif isinstance(state, NonAmplifiedFrozenPBandInvDPAdamWState):
    optimizer_step = _concrete_step(state.step, "step")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    if int(current_step) != optimizer_step:
      raise ValueError("current_step must equal state.step")
    warmup_mode = getattr(state, "warmup_mode", "iid")
    if warmup_mode == "global_correlated":
      if noise_step != int(current_step):
        raise ValueError(
            "global_correlated noise_state.step must equal state.step"
        )
    elif warmup_mode == "iid":
      expected_noise_step = max(0, int(current_step) - state.switch_step)
      if noise_step != expected_noise_step:
        raise ValueError(
            "noise_state.step must equal the post-switch Phase-II step"
        )
    else:
      raise ValueError("frozen-p state has an invalid warmup_mode")
  elif isinstance(state, SegmentedBandInvDPAdamWState):
    optimizer_step = _concrete_step(state.step, "step")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    segment_start = _concrete_step(state.segment_start, "segment_start")
    segment_index = _concrete_step(state.segment_index, "segment_index")
    if int(current_step) != optimizer_step:
      raise ValueError("current_step must equal state.step")
    if segment_index < 0 or segment_start < 0 or segment_start > optimizer_step:
      raise ValueError("segmented state has an invalid current segment")
    if noise_step != optimizer_step - segment_start:
      raise ValueError(
          "noise_state.step must equal the current segment-local step"
      )
  elif isinstance(state, ShadowJMEBandInvDPAdamWState):
    optimizer_step = _concrete_step(state.step, "step")
    phase = _concrete_step(state.phase, "phase")
    segment_start = _concrete_step(state.segment_start, "segment_start")
    segment_index = _concrete_step(state.segment_index, "segment_index")
    noise_m_step = _concrete_step(state.noise_state_m.step, "noise_state_m.step")
    noise_v_step = _concrete_step(state.noise_state_v.step, "noise_state_v.step")
    if int(current_step) != optimizer_step:
      raise ValueError("current_step must equal state.step")
    if phase not in {0, 1} or segment_start < 0 or segment_start > optimizer_step:
      raise ValueError("shadow-JME state has an invalid phase or segment start")
    if segment_index < 0:
      raise ValueError("shadow-JME state has an invalid segment index")
    if phase == 0:
      if noise_m_step != optimizer_step or noise_v_step != 0:
        raise ValueError("shadow-JME warmup noise states have invalid steps")
    else:
      expected_noise_step = optimizer_step - segment_start
      if noise_m_step != expected_noise_step or noise_v_step != expected_noise_step:
        raise ValueError("shadow-JME noise states have invalid local steps")
  elif isinstance(state, NonAmplifiedDPSGDState):
    momentum_step = _concrete_step(state.momentum_state.step, "momentum_state.step")
    if int(current_step) != momentum_step:
      raise ValueError("current_step must equal momentum_state.step")
  elif isinstance(state, NonAmplifiedDPMuonState):
    optimizer_step = _concrete_step(state.step, "step")
    if int(current_step) != optimizer_step:
      raise ValueError("current_step must equal state.step")
  elif isinstance(state, NonAmplifiedDPAdamWState):
    optimizer_step = _concrete_step(state.step, "step")
    if int(current_step) != optimizer_step:
      raise ValueError("current_step must equal state.step")
  elif isinstance(state, PublicVBandInvAdamWState):
    optimizer_step = _concrete_step(state.step, "step")
    first_moment_step = _concrete_step(
        state.optimizer_state.count, "optimizer_state.count"
    )
    segment_start = _concrete_step(state.segment_start, "segment_start")
    segment_end = _concrete_step(state.segment_end, "segment_end")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    if int(current_step) != optimizer_step or int(current_step) != first_moment_step:
      raise ValueError(
          "current_step must equal state.step and optimizer_state.count"
      )
    if not segment_start <= int(current_step) <= segment_end:
      raise ValueError("current_step must lie within the current segment")
    if noise_step != int(current_step) - segment_start:
      raise ValueError("noise_state.step must equal the current segment-local step")
  else:
    raise TypeError("state must be a supported non-amplified training state")


def save_checkpoint(
    path: str | Path,
    *,
    state: (NonAmplifiedBandInvState | NonAmplifiedBandInvDPMuonState |
            NonAmplifiedBandInvDPAdamWState | NonAmplifiedDPSGDState |
            NonAmplifiedDPMuonState | NonAmplifiedDPAdamWState |
            NonAmplifiedBandInvSTPDPAdamWState |
            PublicVBandInvAdamWState | NonAmplifiedFrozenPBandInvDPAdamWState |
            SegmentedBandInvDPAdamWState | ShadowJMEBandInvDPAdamWState),
    current_step: int,
    experiment_config: dict[str, Any],
    artifact_identifiers: dict[str, str],
) -> Path:
  """Atomically stores model state, step, and public run metadata."""
  _validate_steps(state, current_step)
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "format_version": 1,
      "state": jax.device_get(state),
      "current_step": int(current_step),
      "experiment_config": dict(experiment_config),
      "artifact_identifiers": dict(artifact_identifiers),
  }
  with file_lock(destination):
    with atomic_temporary_path(destination) as temporary:
      with temporary.open("wb") as target:
        pickle.dump(payload, target, protocol=pickle.HIGHEST_PROTOCOL)
        target.flush()
        os.fsync(target.fileno())
      atomic_replace(temporary, destination)
  return destination


def load_checkpoint(path: str | Path) -> dict[str, Any]:
  """Loads and validates a checkpoint before it is eligible for resume."""
  with Path(path).open("rb") as source:
    payload = pickle.load(source)
  if not isinstance(payload, dict) or payload.get("format_version") != 1:
    raise ValueError("unsupported or malformed checkpoint")
  required = {"state", "current_step", "experiment_config", "artifact_identifiers"}
  if not required.issubset(payload):
    raise ValueError("checkpoint is missing required fields")
  # ``jax.device_get`` makes ndarray leaves pickleable.  Restore them to JAX
  # arrays before a resumed streaming noise update uses indexed ``.at``.
  payload["state"] = jax.tree_util.tree_map(jax.device_put, payload["state"])
  _validate_steps(payload["state"], payload["current_step"])
  if not isinstance(payload["experiment_config"], dict) or not isinstance(payload["artifact_identifiers"], dict):
    raise ValueError("checkpoint public metadata must be dictionaries")
  return payload


def validate_resume_identity(
    saved: dict[str, Any],
    *,
    experiment_config: dict[str, Any],
    artifact_identifiers: dict[str, str],
) -> None:
  """Rejects resume when any public trajectory identity has changed."""
  if saved["artifact_identifiers"] != dict(artifact_identifiers):
    raise ValueError("checkpoint artifact identifiers do not match this run")
  saved_config = dict(saved["experiment_config"])
  expected_config = dict(experiment_config)
  # The original frozen-p IID checkpoints did not record the newly optional
  # mode field. Treat a missing field as the historical default while still
  # rejecting a checkpoint made for global_correlated.
  if "warmup_mode" in saved_config or "warmup_mode" in expected_config:
    saved_config.setdefault("warmup_mode", "iid")
    expected_config.setdefault("warmup_mode", "iid")
  if saved_config != expected_config:
    raise ValueError("checkpoint experiment config does not match this run")


__all__ = ["load_checkpoint", "save_checkpoint", "validate_resume_identity"]

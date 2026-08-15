"""Checkpointing for complete non-amplified training state."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import jax
import numpy as np

from .nonamplified_dpsgd import NonAmplifiedDPSGDState
from .nonamplified_dpmuon import NonAmplifiedDPMuonState
from .nonamplified_linear import NonAmplifiedBandInvState


def _concrete_step(value: Any, name: str) -> int:
  array = np.asarray(jax.device_get(value))
  if array.shape != () or not np.issubdtype(array.dtype, np.integer):
    raise ValueError(f"{name} must be a concrete integer scalar")
  return int(array)


def _validate_steps(
    state: NonAmplifiedBandInvState | NonAmplifiedDPSGDState | NonAmplifiedDPMuonState, current_step: int
) -> None:
  if not isinstance(current_step, (int, np.integer)) or current_step < 0:
    raise ValueError("current_step must be a non-negative integer")
  if isinstance(state, NonAmplifiedBandInvState):
    nesterov_step = _concrete_step(state.nesterov_state.step, "nesterov_state.step")
    noise_step = _concrete_step(state.noise_state.step, "noise_state.step")
    if int(current_step) != nesterov_step or int(current_step) != noise_step:
      raise ValueError("current_step must equal nesterov_state.step and noise_state.step")
  elif isinstance(state, NonAmplifiedDPSGDState):
    momentum_step = _concrete_step(state.momentum_state.step, "momentum_state.step")
    if int(current_step) != momentum_step:
      raise ValueError("current_step must equal momentum_state.step")
  elif isinstance(state, NonAmplifiedDPMuonState):
    optimizer_step = _concrete_step(state.step, "step")
    if int(current_step) != optimizer_step:
      raise ValueError("current_step must equal state.step")
  else:
    raise TypeError("state must be a supported non-amplified training state")


def save_checkpoint(
    path: str | Path,
    *,
    state: NonAmplifiedBandInvState | NonAmplifiedDPSGDState | NonAmplifiedDPMuonState,
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
  temporary = destination.with_name(destination.name + ".tmp")
  with temporary.open("wb") as target:
    pickle.dump(payload, target, protocol=pickle.HIGHEST_PROTOCOL)
  temporary.replace(destination)
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
  _validate_steps(payload["state"], payload["current_step"])
  if not isinstance(payload["experiment_config"], dict) or not isinstance(payload["artifact_identifiers"], dict):
    raise ValueError("checkpoint public metadata must be dictionaries")
  return payload


__all__ = ["load_checkpoint", "save_checkpoint"]

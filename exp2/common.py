"""Shared Experiment 2 configuration and fixed-cycle contract helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from dp_muon.data import load_cifar10
from dp_muon.training.cifar10_experiment import (
    FixedCycleParticipation,
    derive_fixed_cycle_participation,
)
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import (
    Cifar10BandInvDPAdamWExperimentConfig,
    load_cifar10_bandinv_dpadamw_config,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Exp2Contract:
  """Dataset-derived fixed-cycle values shared by replay and training."""

  num_examples: int
  epochs: int
  batch_size: int
  horizon: int
  min_sep: int
  max_participations: int
  effective_epochs: float


def resolve_repo_path(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else ROOT / path


def derive_contract(
    config: Cifar10BandInvDPAdamWExperimentConfig, *, num_examples: int
) -> Exp2Contract:
  participation = derive_fixed_cycle_participation(
      num_examples, config.epochs, config.batch_size
  )
  return Exp2Contract(
      num_examples=num_examples,
      epochs=config.epochs,
      batch_size=config.batch_size,
      horizon=participation.horizon,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      effective_epochs=participation.effective_epochs,
  )


def load_config_and_contract(
    config_path: str | Path,
) -> tuple[Cifar10BandInvDPAdamWExperimentConfig, Exp2Contract]:
  config_path = resolve_repo_path(config_path)
  config = load_cifar10_bandinv_dpadamw_config(config_path)
  train_images, _ = load_cifar10(config.data_dir, train=True)
  return config, derive_contract(config, num_examples=len(train_images))


def contract_dict(contract: Exp2Contract) -> dict[str, int | float]:
  return asdict(contract)


__all__ = [
    "Exp2Contract",
    "ROOT",
    "contract_dict",
    "derive_contract",
    "load_config_and_contract",
    "resolve_repo_path",
]

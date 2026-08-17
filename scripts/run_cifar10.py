#!/usr/bin/env python3
"""Run a YAML-defined non-amplified CIFAR-10 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dp_muon.training.cifar10_dpsgd_experiment import (
    load_cifar10_dpsgd_momentum_config,
    prepare_cifar10_dpsgd_momentum_run,
    resolve_output_log_dir as resolve_dpsgd_log_dir,
    run_cifar10_dpsgd_momentum,
)
from dp_muon.training.cifar10_dpmuon_experiment import (
    load_cifar10_dpmuon_config,
    prepare_cifar10_dpmuon_run,
    resolve_output_log_dir as resolve_dpmuon_log_dir,
    run_cifar10_dpmuon,
)
from dp_muon.training.cifar10_dpadamw_experiment import (
    load_cifar10_dpadamw_config,
    prepare_cifar10_dpadamw_run,
    resolve_output_log_dir as resolve_dpadamw_log_dir,
    run_cifar10_dpadamw,
)
from dp_muon.training.cifar10_bandinv_dpmuon_experiment import (
    load_cifar10_bandinv_dpmuon_config,
    prepare_cifar10_bandinv_dpmuon_run,
    resolve_output_log_dir as resolve_bandinv_dpmuon_log_dir,
    run_cifar10_bandinv_dpmuon,
)

from dp_muon.training.cifar10_experiment import (
    load_cifar10_nonamplified_config,
    prepare_cifar10_nonamplified_run,
    resolve_output_log_dir,
    run_cifar10_nonamplified,
)


def _config_algorithm(path: str) -> str:
  """Reads the explicitly declared CIFAR-10 algorithm before schema validation."""
  try:
    with Path(path).open(encoding="utf-8") as stream:
      document = yaml.safe_load(stream)
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {path}") from error
  if not isinstance(document, dict):
    raise ValueError("config must be a mapping with an algorithm field")
  algorithm = document.get("algorithm")
  if not isinstance(algorithm, str):
    raise ValueError(
        "config.algorithm is required and must be one of: bandinv, dpsgd, "
        "dpmuon, dp-muon-correlated-naive"
    )
  if algorithm not in {"bandinv", "dpsgd", "dpmuon", "dpadamw", "dp-muon-correlated-naive"}:
    raise ValueError(
        f"unknown config.algorithm {algorithm!r}; expected: bandinv, dpsgd, "
        "dpmuon, dpadamw, dp-muon-correlated-naive"
    )
  return algorithm


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", required=True)
  parser.add_argument("--resume-checkpoint")
  parser.add_argument("--run-dir")
  output = parser.add_mutually_exclusive_group()
  output.add_argument("--print-log-dir", action="store_true")
  output.add_argument("--print-gpu", action="store_true")
  output.add_argument("--prepare-run", action="store_true")
  args = parser.parse_args()
  algorithm = _config_algorithm(args.config)
  if args.print_log_dir:
    print(
        resolve_bandinv_dpmuon_log_dir(args.config)
        if algorithm == "dp-muon-correlated-naive"
        else resolve_dpadamw_log_dir(args.config) if algorithm == "dpadamw"
        else resolve_dpmuon_log_dir(args.config) if algorithm == "dpmuon"
        else resolve_dpsgd_log_dir(args.config) if algorithm == "dpsgd"
        else resolve_output_log_dir(args.config)
    )
    return
  if args.print_gpu:
    config = (
        load_cifar10_bandinv_dpmuon_config(args.config)
        if algorithm == "dp-muon-correlated-naive"
        else load_cifar10_dpadamw_config(args.config) if algorithm == "dpadamw"
        else load_cifar10_dpmuon_config(args.config) if algorithm == "dpmuon"
        else load_cifar10_dpsgd_momentum_config(args.config) if algorithm == "dpsgd"
        else load_cifar10_nonamplified_config(args.config)
    )
    print(config.gpu)
    return
  if args.prepare_run:
    paths = (
        prepare_cifar10_bandinv_dpmuon_run(args.config)
        if algorithm == "dp-muon-correlated-naive"
        else prepare_cifar10_dpadamw_run(args.config) if algorithm == "dpadamw"
        else prepare_cifar10_dpmuon_run(args.config) if algorithm == "dpmuon"
        else prepare_cifar10_dpsgd_momentum_run(args.config) if algorithm == "dpsgd"
        else prepare_cifar10_nonamplified_run(args.config)
    )
    print(paths.directory)
    return
  if algorithm == "dp-muon-correlated-naive":
    if args.resume_checkpoint is None and args.run_dir is None:
      run_cifar10_bandinv_dpmuon(args.config)
    else:
      run_cifar10_bandinv_dpmuon(
          args.config, resume_checkpoint=args.resume_checkpoint, run_dir=args.run_dir
      )
  elif algorithm == "dpadamw":
    if args.resume_checkpoint is None and args.run_dir is None:
      run_cifar10_dpadamw(args.config)
    else:
      run_cifar10_dpadamw(
          args.config, resume_checkpoint=args.resume_checkpoint, run_dir=args.run_dir
      )
  elif algorithm == "dpmuon":
    if args.resume_checkpoint is None and args.run_dir is None:
      run_cifar10_dpmuon(args.config)
    else:
      run_cifar10_dpmuon(
          args.config, resume_checkpoint=args.resume_checkpoint, run_dir=args.run_dir
      )
  elif algorithm == "dpsgd":
    if args.resume_checkpoint is None and args.run_dir is None:
      run_cifar10_dpsgd_momentum(args.config)
    else:
      run_cifar10_dpsgd_momentum(
          args.config,
          resume_checkpoint=args.resume_checkpoint,
          run_dir=args.run_dir,
      )
  else:
    if args.resume_checkpoint is None and args.run_dir is None:
      run_cifar10_nonamplified(args.config)
    else:
      run_cifar10_nonamplified(
          args.config,
          resume_checkpoint=args.resume_checkpoint,
          run_dir=args.run_dir,
      )


if __name__ == "__main__":
  main()

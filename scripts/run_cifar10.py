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

from dp_muon.training.cifar10_experiment import (
    load_cifar10_nonamplified_config,
    prepare_cifar10_nonamplified_run,
    resolve_output_log_dir,
    run_cifar10_nonamplified,
)


def _is_dpsgd_config(path: str) -> bool:
  """Identifies the strategy-free IID baseline schema before full validation."""
  try:
    with Path(path).open(encoding="utf-8") as stream:
      document = yaml.safe_load(stream)
  except FileNotFoundError:
    # Keep the legacy entry point lazy: the selected runner remains responsible
    # for reporting a missing or malformed BandInvMF config.
    return False
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {path}") from error
  return isinstance(document, dict) and "strategy" not in document


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
  is_dpsgd = _is_dpsgd_config(args.config)
  if args.print_log_dir:
    print(resolve_dpsgd_log_dir(args.config) if is_dpsgd else resolve_output_log_dir(args.config))
    return
  if args.print_gpu:
    config = (
        load_cifar10_dpsgd_momentum_config(args.config)
        if is_dpsgd
        else load_cifar10_nonamplified_config(args.config)
    )
    print(config.gpu)
    return
  if args.prepare_run:
    paths = (
        prepare_cifar10_dpsgd_momentum_run(args.config)
        if is_dpsgd
        else prepare_cifar10_nonamplified_run(args.config)
    )
    print(paths.directory)
    return
  if is_dpsgd:
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

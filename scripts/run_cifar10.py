#!/usr/bin/env python3
"""Run the YAML-defined non-amplified BandInvMF CIFAR-10 experiment."""

from __future__ import annotations

import argparse

from dp_muon.training.cifar10_experiment import (
    load_cifar10_nonamplified_config,
    resolve_output_log_dir,
    run_cifar10_nonamplified,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", required=True)
  output = parser.add_mutually_exclusive_group()
  output.add_argument("--print-log-dir", action="store_true")
  output.add_argument("--print-gpu", action="store_true")
  args = parser.parse_args()
  if args.print_log_dir:
    print(resolve_output_log_dir(args.config))
    return
  if args.print_gpu:
    print(load_cifar10_nonamplified_config(args.config).gpu)
    return
  run_cifar10_nonamplified(args.config)


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Run the YAML-defined non-amplified BandInvMF CIFAR-10 experiment."""

from __future__ import annotations

import argparse

from dp_muon.training.cifar10_experiment import run_cifar10_nonamplified


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", required=True)
  args = parser.parse_args()
  _, history = run_cifar10_nonamplified(args.config)
  for record in history:
    print(record)


if __name__ == "__main__":
  main()

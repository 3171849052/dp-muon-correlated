#!/usr/bin/env python3
"""Run the non-amplified IID DP-Muon CIFAR-10 baseline."""

from __future__ import annotations

import argparse

from dp_muon.training.cifar10_dpmuon_experiment import run_cifar10_dpmuon


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="config/cifar10_dpmuon.yaml")
  parser.add_argument("--resume-checkpoint")
  parser.add_argument("--run-dir")
  args = parser.parse_args()
  run_cifar10_dpmuon(args.config, resume_checkpoint=args.resume_checkpoint, run_dir=args.run_dir)


if __name__ == "__main__":
  main()

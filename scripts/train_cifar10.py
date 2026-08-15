#!/usr/bin/env python3
"""CLI entry point for non-amplified BandInvMF CIFAR-10 ViT fine-tuning."""

from __future__ import annotations

import argparse

from dp_muon.training.cifar10_driver import Cifar10TrainConfig, train_cifar10


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--strategy", required=True)
  parser.add_argument("--pretrained", required=True)
  parser.add_argument("--data-dir", required=True)
  parser.add_argument("--batch-size", required=True, type=int)
  parser.add_argument("--microbatch-size", type=int)
  parser.add_argument("--clip-norm", required=True, type=float)
  parser.add_argument("--epsilon", required=True, type=float)
  parser.add_argument("--delta", required=True, type=float)
  parser.add_argument("--momentum", required=True, type=float)
  parser.add_argument("--learning-rate", required=True, type=float)
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--checkpoint-dir", required=True)
  parser.add_argument("--eval-every", required=True, type=int)
  parser.add_argument("--resume-checkpoint")
  args = parser.parse_args()
  config = Cifar10TrainConfig(
      strategy=args.strategy, pretrained=args.pretrained, data_dir=args.data_dir,
      batch_size=args.batch_size, microbatch_size=args.microbatch_size,
      clip_norm=args.clip_norm, epsilon=args.epsilon, delta=args.delta,
      momentum=args.momentum, learning_rate=args.learning_rate, seed=args.seed,
      checkpoint_dir=args.checkpoint_dir, eval_every=args.eval_every,
  )
  _, history = train_cifar10(config, resume_checkpoint=args.resume_checkpoint)
  for record in history:
    print(record)


if __name__ == "__main__":
  main()

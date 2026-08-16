"""Immutable public pretrained-model snapshots for all CIFAR trainers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax

from dp_muon.models import load_pretrained_vit_tiny

from .file_locking import file_fingerprint, file_lock


@dataclass(frozen=True)
class LoadedPretrainedSnapshot:
  """Parameters and identity loaded while the pretrained artifact was locked."""

  params: dict
  path: Path
  sha256: str


def load_pretrained_snapshot(
    path: str | Path, *, key: jax.Array
) -> LoadedPretrainedSnapshot:
  """Loads parameters and hashes exactly the same locked pretrained version."""
  resolved = Path(path).resolve()
  with file_lock(resolved):
    params = load_pretrained_vit_tiny(resolved, key=key)
    return LoadedPretrainedSnapshot(
        params=params, path=resolved, sha256=file_fingerprint(resolved)
    )


__all__ = ["LoadedPretrainedSnapshot", "load_pretrained_snapshot"]

"""CIFAR-10 loading and ViT-compatible logical batch preprocessing."""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from tqdm.auto import tqdm


# Google's public JAX ViT checkpoints are trained with pixels in [-1, 1].
CIFAR10_MEAN = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
CIFAR10_STD = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _dataset_root(data_dir: str | Path) -> Path:
  return Path(data_dir) / "cifar-10-batches-py"


def _ensure_cifar10(data_dir: str | Path, download: bool) -> Path:
  root = _dataset_root(data_dir)
  if root.is_dir():
    return root
  if not download:
    raise FileNotFoundError(f"CIFAR-10 not found at {root}; pass download=True to fetch it")
  directory = Path(data_dir)
  directory.mkdir(parents=True, exist_ok=True)
  archive = directory / "cifar-10-python.tar.gz"
  with tqdm(
      desc="Downloading CIFAR-10", unit="B", unit_scale=True, unit_divisor=1024
  ) as progress:
    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
      if total_size > 0:
        progress.total = total_size
      progress.update(block_count * block_size - progress.n)

    urlretrieve(_CIFAR10_URL, archive, reporthook=reporthook)  # nosec B310 -- fixed public CIFAR source
  with tarfile.open(archive, "r:gz") as source:
    source.extractall(directory, filter="data")
  return root


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
  with path.open("rb") as source:
    values = pickle.load(source, encoding="bytes")
  data = np.asarray(values[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
  labels = np.asarray(values[b"labels"], dtype=np.int32)
  return data, labels


def load_cifar10(data_dir: str | Path, *, train: bool, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
  """Returns raw uint8 NHWC images and int32 labels from CIFAR-10."""
  root = _ensure_cifar10(data_dir, download)
  files = [root / f"data_batch_{index}" for index in range(1, 6)] if train else [root / "test_batch"]
  split = "training" if train else "test"
  batches = [
      _read_batch(path)
      for path in tqdm(files, desc=f"Loading CIFAR-10 {split}", unit="file")
  ]
  images, labels = zip(*batches, strict=True)
  return np.concatenate(images), np.concatenate(labels)


def prepare_cifar10_batch(images: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
  """Resizes one logical batch and returns the exact JAX-friendly schema."""
  images = np.asarray(images)
  labels = np.asarray(labels)
  if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
    raise ValueError("CIFAR images must have shape [B, 32, 32, 3]")
  if labels.shape != (images.shape[0],):
    raise ValueError("labels must have shape [B]")
  try:
    from PIL import Image
  except ImportError as error:  # pragma: no cover - PIL is a runtime dependency for real data only.
    raise ImportError("Pillow is required to resize CIFAR-10") from error
  resized = np.stack([
      np.asarray(Image.fromarray(image).resize((128, 128), Image.Resampling.BICUBIC))
      for image in images
  ]).astype(np.float32)
  normalized = (resized / 255.0 - CIFAR10_MEAN) / CIFAR10_STD
  return {"image": normalized.astype(np.float32), "label": labels.astype(np.int32)}


def iter_logical_batches(
    images: np.ndarray, labels: np.ndarray, schedule: list[np.ndarray] | tuple[np.ndarray, ...]
):
  """Yields logical batches only; it intentionally has no microbatch concept."""
  for indices in schedule:
    indices = np.asarray(indices)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(images)):
      raise ValueError("logical batch indices must be a valid one-dimensional dataset index array")
    yield prepare_cifar10_batch(images[indices], labels[indices])


__all__ = ["CIFAR10_MEAN", "CIFAR10_STD", "iter_logical_batches", "load_cifar10", "prepare_cifar10_batch"]

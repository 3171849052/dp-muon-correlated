"""Private CIFAR-10 and public CIFAR dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
import tarfile
from pathlib import Path
from typing import Literal, Sequence
from urllib.request import urlretrieve

import numpy as np
from tqdm.auto import tqdm

from .cifar10 import load_cifar10


PublicSource = Literal["cifar10_split", "cifar100_10class"]

# Canonical CIFAR-100 fine-label IDs.  These classes have no direct CIFAR-10
# counterpart and their order is also the fixed 0..9 remapping order.
DEFAULT_CIFAR100_PUBLIC_CLASSES: tuple[int, ...] = (
    0,   # apple
    5,   # bed
    10,  # bowl
    12,  # bridge
    17,  # castle
    22,  # clock
    25,  # couch
    28,  # cup
    40,  # lamp
    94,  # wardrobe
)
CIFAR100_FINE_CLASS_NAMES: tuple[str, ...] = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)

_CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


@dataclass(frozen=True)
class PublicPrivateCifarData:
  """Raw datasets and source indices, retained for split verification."""

  private_images: np.ndarray
  private_labels: np.ndarray
  public_images: np.ndarray
  public_labels: np.ndarray
  private_source_indices: np.ndarray
  public_source_indices: np.ndarray
  public_source: PublicSource


def split_cifar10_public_private(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    public_size: int,
    split_seed: int,
) -> PublicPrivateCifarData:
  """Returns a deterministic, exhaustive, disjoint CIFAR-10 split."""
  images, labels = np.asarray(images), np.asarray(labels)
  if labels.shape != (len(images),):
    raise ValueError("CIFAR-10 labels must have shape (num_examples,)")
  if isinstance(public_size, bool) or not isinstance(public_size, int):
    raise ValueError("cifar10_public_size must be an integer")
  if public_size < 1 or public_size >= len(images):
    raise ValueError("cifar10_public_size must be in [1, num_examples - 1]")
  permutation = np.random.default_rng(split_seed).permutation(len(images))
  public_indices = np.sort(permutation[:public_size]).astype(np.int32)
  private_indices = np.sort(permutation[public_size:]).astype(np.int32)
  if np.intersect1d(private_indices, public_indices).size:
    raise AssertionError("private/public CIFAR-10 indices must be disjoint")
  return PublicPrivateCifarData(
      private_images=images[private_indices],
      private_labels=labels[private_indices].astype(np.int32, copy=False),
      public_images=images[public_indices],
      public_labels=labels[public_indices].astype(np.int32, copy=False),
      private_source_indices=private_indices,
      public_source_indices=public_indices,
      public_source="cifar10_split",
  )


def filter_cifar100_public_classes(
    images: np.ndarray,
    fine_labels: np.ndarray,
    public_classes: Sequence[int] = DEFAULT_CIFAR100_PUBLIC_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Filters ten fine classes and remaps configured order to labels 0..9."""
  images, fine_labels = np.asarray(images), np.asarray(fine_labels)
  classes = tuple(int(value) for value in public_classes)
  if (
      len(classes) != 10
      or len(set(classes)) != 10
      or any(value < 0 or value >= 100 for value in classes)
  ):
    raise ValueError("cifar100_public_classes must contain 10 unique IDs in [0, 99]")
  if fine_labels.shape != (len(images),):
    raise ValueError("CIFAR-100 fine labels must have shape (num_examples,)")
  mapping = np.full(100, -1, dtype=np.int32)
  mapping[np.asarray(classes)] = np.arange(10, dtype=np.int32)
  selected = mapping[fine_labels.astype(np.int32, copy=False)] >= 0
  indices = np.flatnonzero(selected).astype(np.int32)
  return images[indices], mapping[fine_labels[indices]], indices


def _ensure_cifar100(data_dir: str | Path, download: bool) -> Path:
  root = Path(data_dir) / "cifar-100-python"
  if root.is_dir():
    return root
  if not download:
    raise FileNotFoundError(f"CIFAR-100 not found at {root}; pass download=True to fetch it")
  directory = Path(data_dir)
  directory.mkdir(parents=True, exist_ok=True)
  archive = directory / "cifar-100-python.tar.gz"
  with tqdm(desc="Downloading CIFAR-100", unit="B", unit_scale=True, unit_divisor=1024) as progress:
    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
      if total_size > 0:
        progress.total = total_size
      progress.update(block_count * block_size - progress.n)

    urlretrieve(_CIFAR100_URL, archive, reporthook=reporthook)  # nosec B310 -- fixed source
  with tarfile.open(archive, "r:gz") as source:
    source.extractall(directory, filter="data")
  return root


def load_cifar100(
    data_dir: str | Path, *, train: bool, download: bool = True
) -> tuple[np.ndarray, np.ndarray]:
  """Returns raw uint8 NHWC images and canonical int32 fine labels."""
  root = _ensure_cifar100(data_dir, download)
  with (root / ("train" if train else "test")).open("rb") as source:
    values = pickle.load(source, encoding="bytes")
  images = (
      np.asarray(values[b"data"], dtype=np.uint8)
      .reshape(-1, 3, 32, 32)
      .transpose(0, 2, 3, 1)
  )
  return images, np.asarray(values[b"fine_labels"], dtype=np.int32)


def load_public_private_cifar(
    data_dir: str | Path,
    *,
    public_source: PublicSource,
    cifar10_public_size: int,
    public_split_seed: int,
    cifar100_public_classes: Sequence[int] = DEFAULT_CIFAR100_PUBLIC_CLASSES,
    download: bool = True,
) -> PublicPrivateCifarData:
  """Loads the common representation consumed by the shared training flow."""
  cifar10_images, cifar10_labels = load_cifar10(data_dir, train=True, download=download)
  if public_source == "cifar10_split":
    return split_cifar10_public_private(
        cifar10_images,
        cifar10_labels,
        public_size=cifar10_public_size,
        split_seed=public_split_seed,
    )
  if public_source != "cifar100_10class":
    raise ValueError("public_source must be 'cifar10_split' or 'cifar100_10class'")
  public_images, fine_labels = load_cifar100(data_dir, train=True, download=download)
  public_images, public_labels, public_indices = filter_cifar100_public_classes(
      public_images, fine_labels, cifar100_public_classes
  )
  return PublicPrivateCifarData(
      private_images=cifar10_images,
      private_labels=cifar10_labels,
      public_images=public_images,
      public_labels=public_labels,
      private_source_indices=np.arange(len(cifar10_images), dtype=np.int32),
      public_source_indices=public_indices,
      public_source="cifar100_10class",
  )


__all__ = [
    "CIFAR100_FINE_CLASS_NAMES",
    "DEFAULT_CIFAR100_PUBLIC_CLASSES",
    "PublicPrivateCifarData",
    "PublicSource",
    "filter_cifar100_public_classes",
    "load_cifar100",
    "load_public_private_cifar",
    "split_cifar10_public_private",
]

import pickle

import numpy as np

from dp_muon.data.cifar10 import load_cifar10


def _write_batch(path, *, count: int, offset: int) -> None:
  values = {
      b"data": (np.arange(count * 3 * 32 * 32, dtype=np.uint32) + offset).astype(np.uint8).reshape(count, -1),
      b"labels": list(range(offset, offset + count)),
  }
  with path.open("wb") as target:
    pickle.dump(values, target)


def test_load_cifar10_unpacks_train_and_test_pickle_splits(tmp_path):
  root = tmp_path / "cifar-10-batches-py"
  root.mkdir()
  for index in range(1, 6):
    _write_batch(root / f"data_batch_{index}", count=2, offset=2 * (index - 1))
  _write_batch(root / "test_batch", count=3, offset=20)

  train_images, train_labels = load_cifar10(tmp_path, train=True, download=False)
  test_images, test_labels = load_cifar10(tmp_path, train=False, download=False)

  assert train_images.shape == (10, 32, 32, 3)
  assert train_labels.shape == (10,)
  assert train_images.dtype == np.uint8
  assert train_labels.dtype == np.int32
  assert test_images.shape == (3, 32, 32, 3)
  assert test_labels.shape == (3,)
  assert test_images.dtype == np.uint8
  assert test_labels.dtype == np.int32

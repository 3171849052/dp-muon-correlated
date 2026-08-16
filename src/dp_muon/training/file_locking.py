"""Small process-safe file publication primitives used by every CIFAR runner."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterator

import fcntl


@contextmanager
def file_lock(target: str | Path) -> Iterator[None]:
  """Serializes writers for one target using its stable ``.lock`` sidecar."""
  lock_path = Path(f"{Path(target)}.lock")
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  with lock_path.open("a+") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def atomic_temporary_path(destination: str | Path) -> Iterator[Path]:
  """Yields a unique sibling temporary path and removes it on failure."""
  final = Path(destination)
  final.parent.mkdir(parents=True, exist_ok=True)
  descriptor, raw_path = tempfile.mkstemp(
      prefix=f".{final.name}.", suffix=".tmp", dir=final.parent
  )
  os.close(descriptor)
  temporary = Path(raw_path)
  try:
    yield temporary
  finally:
    temporary.unlink(missing_ok=True)


def fsync_path(path: str | Path) -> None:
  """Flushes file contents to disk before an atomic publication."""
  with Path(path).open("rb") as source:
    os.fsync(source.fileno())


def atomic_replace(temporary: str | Path, destination: str | Path) -> Path:
  """Durably publishes a completed sibling temporary file."""
  temporary_path = Path(temporary)
  destination_path = Path(destination)
  fsync_path(temporary_path)
  os.replace(temporary_path, destination_path)
  directory_fd = os.open(destination_path.parent, os.O_DIRECTORY)
  try:
    os.fsync(directory_fd)
  finally:
    os.close(directory_fd)
  return destination_path


def atomic_write_text(destination: str | Path, content: str) -> Path:
  """Writes text with a unique temp file and an atomic replacement."""
  with atomic_temporary_path(destination) as temporary:
    with temporary.open("w", encoding="utf-8") as target:
      target.write(content)
      target.flush()
      os.fsync(target.fileno())
    return atomic_replace(temporary, destination)


def file_fingerprint(path: str | Path) -> str:
  """Returns the SHA-256 of actual file bytes, rejecting missing artifacts."""
  source = Path(path)
  if not source.is_file():
    raise ValueError(f"required identity file does not exist: {source}")
  digest = hashlib.sha256()
  with source.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


__all__ = [
    "atomic_replace",
    "atomic_temporary_path",
    "atomic_write_text",
    "file_fingerprint",
    "file_lock",
]

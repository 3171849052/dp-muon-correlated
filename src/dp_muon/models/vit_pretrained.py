"""Conversion of Google's public JAX ViT-Ti/16 NPZ checkpoints."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm

from .vit_tiny import ViTTinyConfig, init_vit_tiny


def _archive(path: str | Path):
  source = str(path)
  if urlparse(source).scheme in {"http", "https"}:
    with urlopen(source) as response:  # nosec B310 -- explicit public checkpoint input
      content_length = response.headers.get("Content-Length")
      total = int(content_length) if content_length is not None else None
      chunks = []
      with tqdm(
          total=total,
          desc="Downloading pretrained ViT",
          unit="B",
          unit_scale=True,
          unit_divisor=1024,
      ) as progress:
        while chunk := response.read(1024 * 1024):
          chunks.append(chunk)
          progress.update(len(chunk))
      return np.load(BytesIO(b"".join(chunks)), allow_pickle=False)
  return np.load(Path(path), allow_pickle=False)


def _require(archive: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...] | None = None) -> jax.Array:
  if name not in archive.files:
    raise ValueError(f"pretrained ViT checkpoint is missing {name!r}")
  value = np.asarray(archive[name])
  if not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
    raise ValueError(f"pretrained checkpoint field {name!r} must be finite floating data")
  if shape is not None and value.shape != shape:
    raise ValueError(f"pretrained checkpoint field {name!r} has shape {value.shape}, expected {shape}")
  return jnp.asarray(value, dtype=jnp.float32)


def interpolate_positional_embedding(position: jax.Array, target_grid: int) -> jax.Array:
  """Matches Google's ``interpolate_posembed`` for the patch position grid."""
  position = np.asarray(position, dtype=np.float32)
  if position.ndim != 3 or position.shape[0] != 1 or position.shape[1] < 2:
    raise ValueError("positional embedding must have shape [1, 1 + grid**2, embed_dim]")
  source_tokens = position.shape[1] - 1
  source_grid = int(round(source_tokens**0.5))
  if source_grid * source_grid != source_tokens:
    raise ValueError("pretrained positional embedding patch tokens must form a square grid")
  cls, patches = position[:, :1], position[:, 1:]
  patches = patches.reshape(source_grid, source_grid, position.shape[-1])
  zoom = (target_grid / source_grid, target_grid / source_grid, 1)
  patches = ndimage.zoom(patches, zoom, order=1)
  resized = np.concatenate(
      (cls, patches.reshape(1, target_grid * target_grid, position.shape[-1])), axis=1
  )
  return jnp.asarray(resized, dtype=jnp.float32)


def _load_qkv_projection(
    archive: np.lib.npyio.NpzFile, prefix: str, projection: str, config: ViTTinyConfig
) -> dict[str, jax.Array]:
  """Converts Flax DenseGeneral Q/K/V tensors to this model's dense layout."""
  head_dim = config.embed_dim // config.num_heads
  base = f"{prefix}/MultiHeadDotProductAttention_1/{projection}"
  return {
      "kernel": _require(
          archive, f"{base}/kernel", (config.embed_dim, config.num_heads, head_dim)
      ).reshape(config.embed_dim, config.embed_dim),
      "bias": _require(
          archive, f"{base}/bias", (config.num_heads, head_dim)
      ).reshape(config.embed_dim),
  }


def _load_out_projection(
    archive: np.lib.npyio.NpzFile, prefix: str, config: ViTTinyConfig
) -> dict[str, jax.Array]:
  """Converts Flax DenseGeneral attention output weights to a dense matrix."""
  head_dim = config.embed_dim // config.num_heads
  base = f"{prefix}/MultiHeadDotProductAttention_1/out"
  return {
      "kernel": _require(
          archive, f"{base}/kernel", (config.num_heads, head_dim, config.embed_dim)
      ).reshape(config.embed_dim, config.embed_dim),
      "bias": _require(archive, f"{base}/bias", (config.embed_dim,)),
  }


def load_pretrained_vit_tiny(
    path: str | Path,
    *,
    key: jax.Array,
    config: ViTTinyConfig = ViTTinyConfig(),
) -> dict:
  """Loads ViT-Ti/16 encoder weights and creates a fresh CIFAR-10 head.

  This accepts the official Google JAX ``.npz`` naming convention.  The
  source classifier is deliberately never read, avoiding class-count coupling.
  """
  if (config.patch_size, config.embed_dim, config.depth, config.num_heads, config.mlp_dim) != (16, 192, 12, 3, 768):
    raise ValueError("Google ViT-Ti/16 checkpoints require the ViT-Tiny/16 architecture")
  params = init_vit_tiny(key, config)
  with _archive(path) as archive:
    params["patch_embedding"] = {
        "kernel": _require(archive, "embedding/kernel", (16, 16, 3, 192)),
        "bias": _require(archive, "embedding/bias", (192,)),
    }
    params["cls"] = _require(archive, "cls", (1, 1, 192))
    params["pos_embedding"] = interpolate_positional_embedding(
        _require(archive, "Transformer/posembed_input/pos_embedding"), config.image_size // config.patch_size
    )
    blocks = []
    for index in range(config.depth):
      prefix = f"Transformer/encoderblock_{index}"
      blocks.append({
          "ln1": {"scale": _require(archive, f"{prefix}/LayerNorm_0/scale", (192,)), "bias": _require(archive, f"{prefix}/LayerNorm_0/bias", (192,))},
          "attention": {
              "query": _load_qkv_projection(archive, prefix, "query", config),
              "key": _load_qkv_projection(archive, prefix, "key", config),
              "value": _load_qkv_projection(archive, prefix, "value", config),
              "out": _load_out_projection(archive, prefix, config),
          },
          "ln2": {"scale": _require(archive, f"{prefix}/LayerNorm_2/scale", (192,)), "bias": _require(archive, f"{prefix}/LayerNorm_2/bias", (192,))},
          "mlp": {
              "dense0": {"kernel": _require(archive, f"{prefix}/MlpBlock_3/Dense_0/kernel", (192, 768)), "bias": _require(archive, f"{prefix}/MlpBlock_3/Dense_0/bias", (768,))},
              "dense1": {"kernel": _require(archive, f"{prefix}/MlpBlock_3/Dense_1/kernel", (768, 192)), "bias": _require(archive, f"{prefix}/MlpBlock_3/Dense_1/bias", (192,))},
          },
      })
    params["blocks"] = tuple(blocks)
    params["encoder_norm"] = {
        "scale": _require(archive, "Transformer/encoder_norm/scale", (192,)),
        "bias": _require(archive, "Transformer/encoder_norm/bias", (192,)),
    }
  return params


__all__ = ["interpolate_positional_embedding", "load_pretrained_vit_tiny"]

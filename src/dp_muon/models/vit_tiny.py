"""A small, framework-free ViT-Tiny/16 classifier for 128px CIFAR inputs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ViTTinyConfig:
  image_size: int = 128
  patch_size: int = 16
  embed_dim: int = 192
  depth: int = 12
  num_heads: int = 3
  mlp_dim: int = 768
  num_classes: int = 10
  layer_norm_epsilon: float = 1e-6

  def __post_init__(self) -> None:
    if self.image_size % self.patch_size:
      raise ValueError("image_size must be divisible by patch_size")
    if self.embed_dim % self.num_heads:
      raise ValueError("embed_dim must be divisible by num_heads")

  @property
  def num_patches(self) -> int:
    return (self.image_size // self.patch_size) ** 2


def _normal(key: jax.Array, shape: tuple[int, ...], scale: float = 0.02) -> jax.Array:
  return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def _dense_params(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
  return {"kernel": _normal(key, (in_dim, out_dim)), "bias": jnp.zeros((out_dim,), jnp.float32)}


def _layer_norm_params(dim: int) -> dict[str, jax.Array]:
  return {"scale": jnp.ones((dim,), jnp.float32), "bias": jnp.zeros((dim,), jnp.float32)}


def init_vit_tiny(key: jax.Array, config: ViTTinyConfig = ViTTinyConfig()) -> dict:
  """Initializes a parameter PyTree; the model definition contains no I/O."""
  keys = iter(jax.random.split(key, 2 + config.depth * 6))
  patch_key, cls_key = next(keys), next(keys)
  blocks = []
  for _ in range(config.depth):
    query_key, key_key, value_key, out_key, mlp0_key, mlp1_key = (next(keys) for _ in range(6))
    blocks.append({
        "ln1": _layer_norm_params(config.embed_dim),
        "attention": {
            "query": _dense_params(query_key, config.embed_dim, config.embed_dim),
            "key": _dense_params(key_key, config.embed_dim, config.embed_dim),
            "value": _dense_params(value_key, config.embed_dim, config.embed_dim),
            "out": _dense_params(out_key, config.embed_dim, config.embed_dim),
        },
        "ln2": _layer_norm_params(config.embed_dim),
        "mlp": {
            "dense0": _dense_params(mlp0_key, config.embed_dim, config.mlp_dim),
            "dense1": _dense_params(mlp1_key, config.mlp_dim, config.embed_dim),
        },
    })
  # A deterministic split keeps the head independent from imported checkpoints.
  head_key = jax.random.fold_in(cls_key, 1)
  return {
      "patch_embedding": {
          "kernel": _normal(patch_key, (config.patch_size, config.patch_size, 3, config.embed_dim)),
          "bias": jnp.zeros((config.embed_dim,), jnp.float32),
      },
      "cls": _normal(cls_key, (1, 1, config.embed_dim)),
      "pos_embedding": _normal(jax.random.fold_in(cls_key, 2), (1, config.num_patches + 1, config.embed_dim)),
      "blocks": tuple(blocks),
      "encoder_norm": _layer_norm_params(config.embed_dim),
      "head": _dense_params(head_key, config.embed_dim, config.num_classes),
  }


def _dense(inputs: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
  return jnp.einsum("...d,df->...f", inputs, params["kernel"]) + params["bias"]


def _layer_norm(inputs: jax.Array, params: dict[str, jax.Array], epsilon: float) -> jax.Array:
  mean = jnp.mean(inputs, axis=-1, keepdims=True)
  variance = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
  return (inputs - mean) * jax.lax.rsqrt(variance + epsilon) * params["scale"] + params["bias"]


def _patchify(images: jax.Array, params: dict[str, jax.Array], config: ViTTinyConfig) -> jax.Array:
  batch, height, width, channels = images.shape
  if (height, width, channels) != (config.image_size, config.image_size, 3):
    raise ValueError(f"images must have shape [B, {config.image_size}, {config.image_size}, 3]")
  grid = config.image_size // config.patch_size
  patches = images.reshape(batch, grid, config.patch_size, grid, config.patch_size, 3)
  patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(batch, grid * grid, -1)
  kernel = params["kernel"].reshape(-1, config.embed_dim)
  return jnp.einsum("bnd,df->bnf", patches, kernel) + params["bias"]


def _attention(inputs: jax.Array, params: dict[str, dict[str, jax.Array]], config: ViTTinyConfig) -> jax.Array:
  q, k, v = (_dense(inputs, params[name]) for name in ("query", "key", "value"))
  batch, tokens, _ = q.shape
  head_dim = config.embed_dim // config.num_heads
  def split_heads(value: jax.Array) -> jax.Array:
    return value.reshape(batch, tokens, config.num_heads, head_dim).transpose(0, 2, 1, 3)
  q, k, v = split_heads(q), split_heads(k), split_heads(v)
  weights = jax.nn.softmax(jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.sqrt(float(head_dim)), axis=-1)
  joined = jnp.einsum("bhij,bhjd->bhid", weights, v).transpose(0, 2, 1, 3)
  return _dense(joined.reshape(batch, tokens, config.embed_dim), params["out"])


def vit_tiny_forward(params: dict, images: jax.Array, config: ViTTinyConfig = ViTTinyConfig()) -> jax.Array:
  """Returns 10-class logits for normalized NHWC images."""
  images = jnp.asarray(images, dtype=jnp.float32)
  tokens = _patchify(images, params["patch_embedding"], config)
  cls = jnp.broadcast_to(params["cls"], (images.shape[0], 1, config.embed_dim))
  tokens = jnp.concatenate((cls, tokens), axis=1) + params["pos_embedding"]
  for block in params["blocks"]:
    tokens = tokens + _attention(_layer_norm(tokens, block["ln1"], config.layer_norm_epsilon), block["attention"], config)
    mlp_input = _layer_norm(tokens, block["ln2"], config.layer_norm_epsilon)
    tokens = tokens + _dense(jax.nn.gelu(_dense(mlp_input, block["mlp"]["dense0"])), block["mlp"]["dense1"])
  return _dense(_layer_norm(tokens[:, 0], params["encoder_norm"], config.layer_norm_epsilon), params["head"])


@dataclass(frozen=True)
class ViTTiny:
  """Tiny convenience wrapper with ``init``/``apply`` methods like JAX modules."""

  config: ViTTinyConfig = ViTTinyConfig()

  def init(self, key: jax.Array) -> dict:
    return init_vit_tiny(key, self.config)

  def apply(self, params: dict, images: jax.Array) -> jax.Array:
    return vit_tiny_forward(params, images, self.config)


__all__ = ["ViTTiny", "ViTTinyConfig", "init_vit_tiny", "vit_tiny_forward"]

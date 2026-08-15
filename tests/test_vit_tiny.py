import jax
import jax.numpy as jnp

from dp_muon.models.vit_tiny import ViTTiny, ViTTinyConfig


def test_vit_tiny_initializes_and_produces_cifar10_logits():
  model = ViTTiny()
  params = model.init(jax.random.key(0))
  assert params["pos_embedding"].shape == (1, 65, 192)
  logits = jax.jit(model.apply)(params, jnp.zeros((1, 128, 128, 3), jnp.float32))
  assert logits.shape == (1, 10)


def test_vit_tiny_configuration_has_required_architecture():
  config = ViTTinyConfig()
  assert (config.image_size, config.patch_size, config.embed_dim, config.depth) == (128, 16, 192, 12)
  assert (config.num_heads, config.mlp_dim, config.num_classes) == (3, 768, 10)

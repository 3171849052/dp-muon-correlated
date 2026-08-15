import jax
import numpy as np

from dp_muon.models import ViTTiny, load_pretrained_vit_tiny


def _checkpoint(path):
  values = {
      "embedding/kernel": np.ones((16, 16, 3, 192), np.float32),
      "embedding/bias": np.ones((192,), np.float32),
      "cls": np.ones((1, 1, 192), np.float32),
      "Transformer/posembed_input/pos_embedding": np.ones((1, 197, 192), np.float32),
      "Transformer/encoder_norm/scale": np.ones((192,), np.float32),
      "Transformer/encoder_norm/bias": np.zeros((192,), np.float32),
  }
  for index in range(12):
    prefix = f"Transformer/encoderblock_{index}"
    for layer in ("LayerNorm_0", "LayerNorm_2"):
      values[f"{prefix}/{layer}/scale"] = np.ones((192,), np.float32)
      values[f"{prefix}/{layer}/bias"] = np.zeros((192,), np.float32)
    for name in ("query", "key", "value", "out"):
      values[f"{prefix}/MultiHeadDotProductAttention_1/{name}/kernel"] = np.eye(192, dtype=np.float32)
      values[f"{prefix}/MultiHeadDotProductAttention_1/{name}/bias"] = np.zeros((192,), np.float32)
    values[f"{prefix}/MlpBlock_3/Dense_0/kernel"] = np.zeros((192, 768), np.float32)
    values[f"{prefix}/MlpBlock_3/Dense_0/bias"] = np.zeros((768,), np.float32)
    values[f"{prefix}/MlpBlock_3/Dense_1/kernel"] = np.zeros((768, 192), np.float32)
    values[f"{prefix}/MlpBlock_3/Dense_1/bias"] = np.zeros((192,), np.float32)
  np.savez(path, **values)


def test_google_npz_loader_imports_encoder_and_replaces_classifier(tmp_path):
  path = tmp_path / "vit_tiny.npz"
  _checkpoint(path)
  params = load_pretrained_vit_tiny(path, key=jax.random.key(1))
  np.testing.assert_allclose(params["patch_embedding"]["kernel"], 1.0)
  np.testing.assert_allclose(params["blocks"][0]["attention"]["query"]["kernel"], np.eye(192))
  assert params["pos_embedding"].shape == (1, 65, 192)
  assert ViTTiny().apply(params, np.zeros((1, 128, 128, 3), np.float32)).shape == (1, 10)

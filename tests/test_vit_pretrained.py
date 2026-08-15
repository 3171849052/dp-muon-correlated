import jax
import numpy as np
import pytest

from dp_muon.models import ViTTiny, load_pretrained_vit_tiny


def _checkpoint(path, *, malformed_query: bool = False):
  values = {
      "embedding/kernel": np.ones((16, 16, 3, 192), np.float32),
      "embedding/bias": np.ones((192,), np.float32),
      "cls": np.ones((1, 1, 192), np.float32),
      "Transformer/posembed_input/pos_embedding": np.ones((1, 197, 192), np.float32),
      "Transformer/encoder_norm/scale": np.ones((192,), np.float32),
      "Transformer/encoder_norm/bias": np.zeros((192,), np.float32),
      # A source classifier is deliberately ignored by the loader.
      "head/kernel": np.full((192, 1000), 7.0, np.float32),
      "head/bias": np.full((1000,), 7.0, np.float32),
  }
  for index in range(12):
    prefix = f"Transformer/encoderblock_{index}"
    for layer in ("LayerNorm_0", "LayerNorm_2"):
      values[f"{prefix}/{layer}/scale"] = np.ones((192,), np.float32)
      values[f"{prefix}/{layer}/bias"] = np.zeros((192,), np.float32)
    for projection in ("query", "key", "value"):
      values[f"{prefix}/MultiHeadDotProductAttention_1/{projection}/kernel"] = np.arange(
          192 * 3 * 64, dtype=np.float32
      ).reshape(192, 3, 64) + index
      values[f"{prefix}/MultiHeadDotProductAttention_1/{projection}/bias"] = np.arange(
          3 * 64, dtype=np.float32
      ).reshape(3, 64) + index
    values[f"{prefix}/MultiHeadDotProductAttention_1/out/kernel"] = np.arange(
        3 * 64 * 192, dtype=np.float32
    ).reshape(3, 64, 192) + index
    values[f"{prefix}/MultiHeadDotProductAttention_1/out/bias"] = np.arange(192, dtype=np.float32) + index
    values[f"{prefix}/MlpBlock_3/Dense_0/kernel"] = np.zeros((192, 768), np.float32)
    values[f"{prefix}/MlpBlock_3/Dense_0/bias"] = np.zeros((768,), np.float32)
    values[f"{prefix}/MlpBlock_3/Dense_1/kernel"] = np.zeros((768, 192), np.float32)
    values[f"{prefix}/MlpBlock_3/Dense_1/bias"] = np.zeros((192,), np.float32)
  if malformed_query:
    values["Transformer/encoderblock_0/MultiHeadDotProductAttention_1/query/kernel"] = np.zeros(
        (192, 192), np.float32
    )
  np.savez(path, **values)


def test_google_npz_loader_imports_densegeneral_encoder_and_replaces_classifier(tmp_path):
  path = tmp_path / "vit_tiny.npz"
  _checkpoint(path)
  params = load_pretrained_vit_tiny(path, key=jax.random.key(1))
  attention = params["blocks"][0]["attention"]
  with np.load(path, allow_pickle=False) as source:
    for name in ("query", "key", "value"):
      kernel = source[f"Transformer/encoderblock_0/MultiHeadDotProductAttention_1/{name}/kernel"]
      bias = source[f"Transformer/encoderblock_0/MultiHeadDotProductAttention_1/{name}/bias"]
      assert attention[name]["kernel"].shape == (192, 192)
      assert attention[name]["bias"].shape == (192,)
      np.testing.assert_array_equal(attention[name]["kernel"], kernel.reshape(192, 192))
      np.testing.assert_array_equal(attention[name]["bias"], bias.reshape(192))
    out_kernel = source["Transformer/encoderblock_0/MultiHeadDotProductAttention_1/out/kernel"]
    assert attention["out"]["kernel"].shape == (192, 192)
    np.testing.assert_array_equal(attention["out"]["kernel"], out_kernel.reshape(192, 192))

  np.testing.assert_allclose(params["patch_embedding"]["kernel"], 1.0)
  assert params["pos_embedding"].shape == (1, 65, 192)
  assert params["head"]["kernel"].shape == (192, 10)
  assert not np.allclose(params["head"]["kernel"], 7.0)
  assert ViTTiny().apply(params, np.zeros((1, 128, 128, 3), np.float32)).shape == (1, 10)


def test_google_npz_loader_rejects_non_densegeneral_attention_shapes(tmp_path):
  path = tmp_path / "malformed_vit_tiny.npz"
  _checkpoint(path, malformed_query=True)
  with pytest.raises(ValueError, match="query/kernel.*expected"):
    load_pretrained_vit_tiny(path, key=jax.random.key(1))

import jax

from dp_muon.models import ViTTiny
from dp_muon.optim import ADAMW, MUON, count_muon_parameters, vit_muon_parameter_labels


def test_vit_tiny_has_exactly_72_muon_block_kernels():
  labels = vit_muon_parameter_labels(ViTTiny().init(jax.random.key(0)))
  assert count_muon_parameters(ViTTiny().init(jax.random.key(0))) == 72
  assert labels["blocks"][0]["attention"]["query"]["kernel"] == MUON
  assert labels["blocks"][11]["mlp"]["dense1"]["kernel"] == MUON


def test_only_block_kernels_are_muon_parameters():
  labels = vit_muon_parameter_labels(ViTTiny().init(jax.random.key(1)))
  assert labels["head"]["kernel"] == ADAMW
  assert labels["patch_embedding"]["kernel"] == ADAMW
  assert labels["cls"] == ADAMW
  assert labels["pos_embedding"] == ADAMW
  assert labels["blocks"][0]["attention"]["query"]["bias"] == ADAMW
  assert labels["blocks"][0]["ln1"]["scale"] == ADAMW
  assert labels["encoder_norm"]["bias"] == ADAMW

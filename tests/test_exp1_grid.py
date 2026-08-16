from collections import Counter
from pathlib import Path
import sys

import pytest
import yaml

from dp_muon.training.cifar10_dpmuon_experiment import load_cifar10_dpmuon_config
from dp_muon.training.cifar10_dpsgd_experiment import load_cifar10_dpsgd_momentum_config
from dp_muon.training.cifar10_experiment import load_cifar10_nonamplified_config
from exp1 import generate_configs
from scripts import run_cifar10


LOADERS = {
    "bandinv": load_cifar10_nonamplified_config,
    "dpsgd": load_cifar10_dpsgd_momentum_config,
    "dpmuon": load_cifar10_dpmuon_config,
}


def _copy_grid_inputs(tmp_path: Path) -> None:
  (tmp_path / "config").mkdir()
  for name in (
      "cifar10_nonamplified.yaml",
      "cifar10_dpsgd_momentum.yaml",
      "cifar10_dpmuon.yaml",
  ):
    (tmp_path / "config" / name).write_text(
        (Path("config") / name).read_text(encoding="utf-8"), encoding="utf-8"
    )
  (tmp_path / "exp1").mkdir()
  (tmp_path / "exp1" / "grid.yaml").write_text(
      Path("exp1/grid.yaml").read_text(encoding="utf-8"), encoding="utf-8"
  )


def test_generator_creates_valid_27_run_grid_and_removes_stale_yaml(tmp_path, monkeypatch):
  _copy_grid_inputs(tmp_path)
  exp1 = tmp_path / "exp1"
  config_root = exp1 / "config"
  stale = config_root / "obsolete.yaml"
  config_root.mkdir()
  stale.write_text("obsolete: true\n", encoding="utf-8")
  monkeypatch.setattr(generate_configs, "ROOT", tmp_path)
  monkeypatch.setattr(generate_configs, "EXP1", exp1)
  monkeypatch.setattr(generate_configs, "GRID_PATH", exp1 / "grid.yaml")
  monkeypatch.setattr(generate_configs, "CONFIG_ROOT", config_root)
  monkeypatch.setattr(generate_configs, "MANIFEST_PATH", exp1 / "manifest.tsv")

  records = generate_configs.generate()

  assert len(records) == 27
  assert not stale.exists()
  assert Counter(record["algorithm"] for record in records) == {
      "bandinv": 9, "dpsgd": 9, "dpmuon": 9,
  }
  assert Counter(record["gpu"] for record in records) == {"0": 9, "1": 9, "2": 9}
  manifest = (exp1 / "manifest.tsv").read_text(encoding="utf-8").splitlines()
  assert manifest[0] == "id\talgorithm\tlr_setting\tclip_norm\tgpu\tconfig"
  assert len(manifest) == 28

  for record in records:
    document = yaml.safe_load((tmp_path / record["config"]).read_text(encoding="utf-8"))
    assert document["algorithm"] == record["algorithm"]
    assert document["experiment"]["name"] == f"exp1_{record['id']}"
    assert document["runtime"]["gpu"] == int(record["gpu"])
    assert document["training"]["clip_norm"] == float(record["clip_norm"])
    assert document["output"]["log_dir"] == f"exp1/logs/{record['algorithm']}"
    LOADERS[record["algorithm"]](tmp_path / record["config"])

  bandinv = yaml.safe_load(
      (config_root / "bandinv" / "lr0.1_clip1.yaml").read_text(encoding="utf-8")
  )
  assert bandinv["output"]["strategy_dir"] == "artifacts/strategies"
  assert bandinv["strategy"]["force_refit"] is False
  dpmuon = yaml.safe_load(
      (config_root / "dpmuon" / "base_clip5.yaml").read_text(encoding="utf-8")
  )
  assert dpmuon["muon"]["learning_rate"] == 0.02
  assert dpmuon["adamw"]["learning_rate"] == 0.001


@pytest.mark.parametrize(
    ("algorithm", "config_name", "runner_name"),
    [
        ("bandinv", "cifar10_nonamplified.yaml", "run_cifar10_nonamplified"),
        ("dpsgd", "cifar10_dpsgd_momentum.yaml", "run_cifar10_dpsgd_momentum"),
        ("dpmuon", "cifar10_dpmuon.yaml", "run_cifar10_dpmuon"),
    ],
)
def test_runner_routes_by_explicit_algorithm(
    algorithm, config_name, runner_name, monkeypatch
):
  calls = []
  monkeypatch.setattr(run_cifar10, runner_name, lambda path: calls.append(path))
  monkeypatch.setattr(
      sys,
      "argv",
      ["run_cifar10.py", "--config", str(Path("config") / config_name)],
  )
  run_cifar10.main()
  assert calls == [str(Path("config") / config_name)]


def test_runner_rejects_missing_or_unknown_algorithm(tmp_path):
  missing = tmp_path / "missing.yaml"
  missing.write_text("experiment: {}\n", encoding="utf-8")
  with pytest.raises(ValueError, match="config.algorithm is required"):
    run_cifar10._config_algorithm(str(missing))
  unknown = tmp_path / "unknown.yaml"
  unknown.write_text("algorithm: other\n", encoding="utf-8")
  with pytest.raises(ValueError, match="unknown config.algorithm"):
    run_cifar10._config_algorithm(str(unknown))


@pytest.mark.parametrize(
    ("loader", "config_name", "current_algorithm", "wrong_algorithm"),
    [
        (load_cifar10_nonamplified_config, "cifar10_nonamplified.yaml", "bandinv", "dpsgd"),
        (load_cifar10_dpsgd_momentum_config, "cifar10_dpsgd_momentum.yaml", "dpsgd", "dpmuon"),
        (load_cifar10_dpmuon_config, "cifar10_dpmuon.yaml", "dpmuon", "bandinv"),
    ],
)
def test_loaders_require_their_explicit_algorithm(
    tmp_path, loader, config_name, current_algorithm, wrong_algorithm
):
  config_path = tmp_path / config_name
  config_path.write_text(
      (Path("config") / config_name).read_text(encoding="utf-8").replace(
          f"algorithm: {current_algorithm}",
          f"algorithm: {wrong_algorithm}",
      ),
      encoding="utf-8",
  )
  with pytest.raises(ValueError, match="requires algorithm"):
    loader(config_path)

from types import SimpleNamespace

import numpy as np

from dp_muon.optim import (
    decayed_prefix_sum_workload_coef,
    fixed_lr_nesterov_decayed_trajectory_workload_coef,
    fixed_lr_nesterov_trajectory_workload_coef,
)
from scripts import fit_bandinvmf


def _run_main(monkeypatch, tmp_path, workload, **options):
  captured = {}

  def fake_fit(*args, **kwargs):
    captured["workload"] = kwargs["workload_coef"]
    return SimpleNamespace()

  def fake_publish(output, strategy, **kwargs):
    captured["output"] = output
    captured.update(kwargs)
    return output

  monkeypatch.setattr(fit_bandinvmf, "fit_bandinv_strategy", fake_fit)
  monkeypatch.setattr(fit_bandinvmf, "publish_strategy_artifact", fake_publish)
  argv = [
      "fit_bandinvmf.py", "--horizon", "5", "--bandwidth", "2",
      "--min-sep", "1", "--workload", workload, "--output", str(tmp_path / "x.npz"),
  ]
  for name, value in options.items():
    argv.extend([f"--{name.replace('_', '-')}", str(value)])
  monkeypatch.setattr("sys.argv", argv)
  fit_bandinvmf.main()
  return captured


def test_nesterov_trajectory_keeps_legacy_workload_and_metadata(monkeypatch, tmp_path):
  captured = _run_main(
      monkeypatch, tmp_path, "nesterov-trajectory", momentum=0.7, learning_rate=0.1
  )
  np.testing.assert_array_equal(
      np.asarray(captured["workload"]),
      np.asarray(fixed_lr_nesterov_trajectory_workload_coef(5, 0.7, 0.1)),
  )
  assert captured["workload_type"] == "nesterov-trajectory"


def test_nesterov_decayed_trajectory_uses_decayed_workload(monkeypatch, tmp_path):
  captured = _run_main(
      monkeypatch, tmp_path, "nesterov-decayed-trajectory",
      momentum=0.7, learning_rate=0.1, weight_decay=0.2,
  )
  np.testing.assert_array_equal(
      np.asarray(captured["workload"]),
      np.asarray(fixed_lr_nesterov_decayed_trajectory_workload_coef(5, 0.7, 0.1, 0.2)),
  )
  assert captured["workload_type"] == "nesterov-decayed-trajectory"


def test_prefix_and_decayed_prefix_are_distinct(monkeypatch, tmp_path):
  prefix = _run_main(monkeypatch, tmp_path, "prefix")
  decayed = _run_main(
      monkeypatch, tmp_path, "decayed-prefix", learning_rate=0.1, weight_decay=0.2
  )
  assert prefix["workload"] is None
  np.testing.assert_array_equal(
      np.asarray(decayed["workload"]),
      np.asarray(decayed_prefix_sum_workload_coef(5, 0.1, 0.2)),
  )
  assert prefix["workload_type"] == "prefix"
  assert decayed["workload_type"] == "decayed-prefix-sum"


def test_default_artifact_paths_have_four_workload_names(tmp_path):
  common = dict(root=tmp_path, horizon=5, bandwidth=2, min_sep=1, max_participations=None)
  paths = {
      workload: fit_bandinvmf.default_artifact_path(
          workload=workload, **common, momentum=0.7, learning_rate=0.1, weight_decay=0.2
      )
      for workload in ("prefix", "nesterov-trajectory", "decayed-prefix", "nesterov-decayed-trajectory")
  }
  assert len({path.name for path in paths.values()}) == 4
  assert paths["prefix"].name.startswith("prefix_")
  assert paths["nesterov-trajectory"].name.startswith("nesterov-trajectory_")
  assert paths["decayed-prefix"].name.startswith("decayed-prefix-sum_")
  assert paths["nesterov-decayed-trajectory"].name.startswith("nesterov-decayed-trajectory_")

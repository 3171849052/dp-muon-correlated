"""Cross-process-safe persistence and public resume identity regressions."""

from __future__ import annotations

from pathlib import Path
import pickle
import threading
import time

import jax.numpy as jnp
import pytest

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.training import checkpoint
from dp_muon.training import bandinvmf_strategy_manager as strategies
from dp_muon.training import cifar10_bandinv_dpmuon_experiment as naive
from dp_muon.training import cifar10_experiment as bandinv
from dp_muon.training.file_locking import file_fingerprint
from dp_muon.training.run_logging import MetricsCSVWriter, RunPaths, write_run_configuration


def test_checkpoint_concurrent_atomic_publication_is_always_readable(tmp_path, monkeypatch):
  target = tmp_path / "checkpoints" / "latest.pkl"
  monkeypatch.setattr(checkpoint, "_validate_steps", lambda *args: None)
  barrier = threading.Barrier(2)

  def save(value):
    barrier.wait()
    checkpoint.save_checkpoint(
        target, state={"writer": value}, current_step=0,
        experiment_config={"run": 1}, artifact_identifiers={"algorithm": "dpsgd"},
    )

  workers = [threading.Thread(target=save, args=(value,)) for value in (1, 2)]
  for worker in workers:
    worker.start()
  for worker in workers:
    worker.join()
  with target.open("rb") as source:
    assert pickle.load(source)["state"]["writer"] in {1, 2}


def test_metrics_concurrent_same_step_has_one_complete_row(tmp_path):
  path = tmp_path / "metrics.csv"
  barrier = threading.Barrier(2)
  record = {
      "epoch": 1, "step": 1, "effective_epoch": 1.0, "epsilon_spent": 1.0,
      "test_loss": 1.0, "test_accuracy": 0.5, "elapsed_seconds": 1.0,
      "eval_seconds": 0.1,
  }
  outcomes = []

  def append():
    writer = MetricsCSVWriter(path)
    barrier.wait()
    outcomes.append(writer.append(record))

  workers = [threading.Thread(target=append) for _ in range(2)]
  for worker in workers:
    worker.start()
  for worker in workers:
    worker.join()
  assert sorted(outcomes) == [False, True]
  assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_run_snapshots_are_atomic_under_competing_writers(tmp_path):
  paths = RunPaths(
      directory=tmp_path, config=tmp_path / "config.yaml",
      resolved_config=tmp_path / "resolved_config.yaml", metrics=tmp_path / "metrics.csv",
      train_log=tmp_path / "train.log", checkpoint=tmp_path / "checkpoints/latest.pkl",
  )
  barrier = threading.Barrier(2)

  def write(value):
    barrier.wait()
    write_run_configuration(paths, source_yaml=f"value: {value}\n", resolved={"value": value})

  workers = [threading.Thread(target=write, args=(value,)) for value in (1, 2)]
  for worker in workers:
    worker.start()
  for worker in workers:
    worker.join()
  assert paths.config.read_text(encoding="utf-8") in {"value: 1\n", "value: 2\n"}
  assert paths.resolved_config.read_text(encoding="utf-8") in {"value: 1\n", "value: 2\n"}


def _request(tmp_path, **changes):
  values = dict(
      horizon=4, min_sep=1, max_participations=2, bandwidth=2, momentum=0.9,
      learning_rate=0.1, reduction="mean", max_optimizer_steps=1,
      strategy_dir=tmp_path, force_refit=False,
  )
  values.update(changes)
  return strategies.BandInvMFFitRequest(**values)


def _strategy(request):
  return BandInvMFStrategy(
      horizon=request.horizon, bandwidth=request.bandwidth, min_sep=request.min_sep,
      max_participations=request.max_participations,
      workload_coef=fixed_lr_nesterov_trajectory_workload_coef(
          request.horizon, request.momentum, request.learning_rate
      ),
      noising_coef=jnp.ones((request.bandwidth,)), strategy_coef=jnp.ones((request.horizon,)),
      sensitivity_squared=jnp.asarray(1.0), objective=jnp.asarray(1.0),
  )


def test_strategy_cache_lock_allows_one_fit_then_reuse(tmp_path):
  request = _request(tmp_path)
  calls = []
  barrier = threading.Barrier(2)
  results = []

  def fit(*args, **kwargs):
    calls.append(1)
    time.sleep(0.1)
    return _strategy(request)

  def worker():
    barrier.wait()
    results.append(strategies.get_or_fit_strategy(request, fit_strategy=fit)[2])

  workers = [threading.Thread(target=worker) for _ in range(2)]
  for worker in workers:
    worker.start()
  for worker in workers:
    worker.join()
  assert len(calls) == 1
  assert sorted(results) == ["fit", "reuse"]


def test_strategy_fit_exception_does_not_publish_artifact(tmp_path):
  request = _request(tmp_path)
  with pytest.raises(RuntimeError, match="fit failed"):
    strategies.get_or_fit_strategy(
        request, fit_strategy=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed"))
    )
  path = strategies.strategy_artifact_path(
      request.strategy_dir, horizon=request.horizon, min_sep=request.min_sep,
      max_participations=request.max_participations, bandwidth=request.bandwidth,
      momentum=request.momentum, learning_rate=request.learning_rate,
      reduction=request.reduction, max_optimizer_steps=request.max_optimizer_steps,
  )
  assert not path.exists()
  assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_resume_strategy_requirement_fails_without_refitting(tmp_path):
  request = _request(tmp_path, force_refit=True)
  with pytest.raises(ValueError, match="required compatible"):
    strategies.require_compatible_strategy(request)


def test_bandinvmf_algorithms_use_the_same_shared_manager():
  assert bandinv._get_or_fit_shared_strategy is strategies.get_or_fit_strategy
  assert naive._get_or_fit_shared_strategy is strategies.get_or_fit_strategy


@pytest.mark.parametrize("algorithm", ["bandinv", "dpsgd", "dpmuon", "dp-muon-correlated-naive"])
def test_resume_identity_rejects_changed_public_config_and_artifact(algorithm, tmp_path):
  pretrained = tmp_path / "pretrained.npz"
  pretrained.write_bytes(b"first")
  identifiers = {
      "algorithm": algorithm,
      "pretrained_path": str(pretrained),
      "pretrained_sha256": file_fingerprint(pretrained),
  }
  if algorithm in {"bandinv", "dp-muon-correlated-naive"}:
    strategy = tmp_path / "strategy.npz"
    strategy.write_bytes(b"strategy")
    identifiers.update(strategy_path=str(strategy), strategy_sha256=file_fingerprint(strategy))
  saved = {"experiment_config": {"learning_rate": 1.0}, "artifact_identifiers": identifiers}
  checkpoint.validate_resume_identity(saved, experiment_config={"learning_rate": 1.0}, artifact_identifiers=identifiers)
  with pytest.raises(ValueError, match="experiment config"):
    checkpoint.validate_resume_identity(saved, experiment_config={"learning_rate": 2.0}, artifact_identifiers=identifiers)
  pretrained.write_bytes(b"replaced")
  changed = dict(identifiers, pretrained_sha256=file_fingerprint(pretrained))
  with pytest.raises(ValueError, match="artifact identifiers"):
    checkpoint.validate_resume_identity(saved, experiment_config={"learning_rate": 1.0}, artifact_identifiers=changed)

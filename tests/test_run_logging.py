from datetime import datetime

from dp_muon.training.run_logging import (
    METRICS_FIELDS,
    MetricsCSVWriter,
    config_content_hash,
    create_run_directory,
    write_run_configuration,
)


def test_run_directory_is_unique_and_snapshots_configuration(tmp_path):
  document = {"training": {"learning_rate": 0.5}, "privacy": {"epsilon": 2.0}}
  config_hash = config_content_hash(document)
  now = datetime(2026, 8, 15, 12, 34, 56)
  first = create_run_directory(
      tmp_path, epsilon=2.0, bandwidth=4, learning_rate=0.5,
      clip_norm=1.0, seed=3, config_hash=config_hash, now=now,
  )
  second = create_run_directory(
      tmp_path, epsilon=2.0, bandwidth=4, learning_rate=0.5,
      clip_norm=1.0, seed=3, config_hash=config_hash, now=now,
  )
  assert first.directory != second.directory
  assert first.directory.name.startswith("20260815-123456_eps2.0_bw4_lr0.5_clip1.0_s3_")
  write_run_configuration(first, source_yaml="privacy: {epsilon: 2.0}\n", resolved=document)
  assert first.config.read_text() == "privacy: {epsilon: 2.0}\n"
  assert first.resolved_config.is_file()
  assert first.train_log.is_file()


def test_metrics_writer_has_exact_schema_and_skips_duplicate_step(tmp_path):
  writer = MetricsCSVWriter(tmp_path / "metrics.csv")
  record = dict(zip(METRICS_FIELDS, (1, 4, 1.0, 0.6, 2.0, 0.8, 3.0, 0.2)))
  assert writer.append(record)
  assert not writer.append(record)
  assert (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",") == list(METRICS_FIELDS)

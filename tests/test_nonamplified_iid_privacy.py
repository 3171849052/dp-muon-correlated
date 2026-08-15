import math

import pytest

from dp_muon.privacy import calibrate_nonamplified_iid, compute_query_sensitivity


def test_iid_calibration_scales_noise_by_full_transcript_sensitivity():
  result = calibrate_nonamplified_iid(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=3.0,
      normalize_by=12.0,
      adjacency="add_remove",
      max_participations=9,
  )
  query_sensitivity = compute_query_sensitivity(3.0, 12.0, "add_remove")
  assert result.query_sensitivity == pytest.approx(query_sensitivity)
  assert result.matrix_sensitivity == pytest.approx(math.sqrt(9.0))
  assert result.total_sensitivity == pytest.approx(query_sensitivity * math.sqrt(9.0))
  assert result.iid_noise_std / (query_sensitivity * math.sqrt(9.0)) == pytest.approx(
      result.noise_multiplier
  )
  assert result.max_participations == 9


def test_replace_one_doubles_iid_noise_and_invalid_k_fails_fast():
  common = dict(
      epsilon=2.0, delta=1e-5, clip_norm=1.0, normalize_by=4.0, max_participations=3
  )
  add_remove = calibrate_nonamplified_iid(adjacency="add_remove", **common)
  replace_one = calibrate_nonamplified_iid(adjacency="replace_one", **common)
  assert replace_one.iid_noise_std == pytest.approx(2.0 * add_remove.iid_noise_std)
  with pytest.raises(ValueError, match="max_participations"):
    calibrate_nonamplified_iid(adjacency="add_remove", max_participations=0, **{
        key: value for key, value in common.items() if key != "max_participations"
    })

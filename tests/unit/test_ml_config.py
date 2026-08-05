"""Training hyperparameter config: loaded from YAML, never hardcoded."""

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from marketpulse.ml.config import DEFAULT_CONFIG_PATH, load_training_config

pytestmark = pytest.mark.unit


def test_default_config_file_exists() -> None:
    assert DEFAULT_CONFIG_PATH.is_file()


def test_load_training_config_parses_the_shipped_default() -> None:
    config = load_training_config()
    assert config.config_version >= 1
    assert config.labeling.horizon_minutes > 0
    assert 0 < config.labeling.theta_quantile < 1
    assert config.split.train_frac + config.split.val_frac < 1.0


def test_labeling_horizon_is_a_timedelta() -> None:
    config = load_training_config()
    assert config.labeling.horizon == timedelta(minutes=config.labeling.horizon_minutes)


def test_split_test_frac_is_derived_not_stored() -> None:
    config = load_training_config()
    expected = 1.0 - config.split.train_frac - config.split.val_frac
    assert config.split.test_frac == pytest.approx(expected)


def test_lightgbm_booster_params_excludes_train_only_fields() -> None:
    config = load_training_config()
    params = config.lightgbm.booster_params(num_class=3)
    assert params["num_class"] == 3
    assert "num_boost_round" not in params
    assert "early_stopping_rounds" not in params


def test_rejects_out_of_range_learning_rate(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
config_version: 1
labeling: {horizon_minutes: 15, theta_quantile: 0.4, theta: null}
split: {train_frac: 0.7, val_frac: 0.15, min_rows_per_split: 200, min_class_frac: 0.05}
lightgbm:
  objective: multiclass
  learning_rate: -0.1
  num_leaves: 31
  max_depth: -1
  min_data_in_leaf: 20
  feature_fraction: 0.9
  bagging_fraction: 0.8
  bagging_freq: 5
  lambda_l1: 0.0
  lambda_l2: 0.0
  num_boost_round: 500
  early_stopping_rounds: 30
  seed: 42
evaluation: {random_baseline_seed: 42, time_buckets: 5}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_training_config(bad)


def test_rejects_split_fractions_outside_zero_one(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
config_version: 1
labeling: {horizon_minutes: 15, theta_quantile: 0.4, theta: null}
split: {train_frac: 1.5, val_frac: 0.15, min_rows_per_split: 200, min_class_frac: 0.05}
lightgbm:
  objective: multiclass
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  min_data_in_leaf: 20
  feature_fraction: 0.9
  bagging_fraction: 0.8
  bagging_freq: 5
  lambda_l1: 0.0
  lambda_l2: 0.0
  num_boost_round: 500
  early_stopping_rounds: 30
  seed: 42
evaluation: {random_baseline_seed: 42, time_buckets: 5}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_training_config(bad)

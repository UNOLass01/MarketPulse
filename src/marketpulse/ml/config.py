"""Versioned training hyperparameters, loaded from ``configs/ml/lightgbm.yaml``.

Never hardcode a hyperparameter in ``ml/train.py`` or ``ml/dataset.py`` --
read it from here so every run logs exactly what produced it (CLAUDE.md:
"Hyperparameters from a versioned config file, never hardcoded"). ``H`` and
``theta`` in particular are logged alongside the model because they define
the labeling problem, not just how it's solved.
"""

from datetime import timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# src/marketpulse/ml/config.py -> parents[3] is the repo root, where
# configs/ lives alongside docker/ and docs/ -- resolved from this file's own
# location so it's independent of the process's current working directory.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "ml" / "lightgbm.yaml"


class LabelingConfig(BaseModel):
    """Hyperparameters that define the labeling problem itself."""

    model_config = ConfigDict(frozen=True)

    horizon_minutes: float = Field(gt=0)
    theta_quantile: float = Field(gt=0, lt=1)
    theta: float | None = Field(default=None, ge=0)

    @property
    def horizon(self) -> timedelta:
        return timedelta(minutes=self.horizon_minutes)


class SplitConfig(BaseModel):
    """Chronological train/validation/test split + sufficiency thresholds."""

    model_config = ConfigDict(frozen=True)

    train_frac: float = Field(gt=0, lt=1)
    val_frac: float = Field(gt=0, lt=1)
    min_rows_per_split: int = Field(gt=0)
    min_class_frac: float = Field(ge=0, lt=1)

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.val_frac


class LightGBMConfig(BaseModel):
    """LightGBM booster hyperparameters. No scaler/normaliser field exists on
    purpose -- LightGBM is scale-invariant and this project never fits one.
    """

    model_config = ConfigDict(frozen=True)

    objective: str = "multiclass"
    learning_rate: float = Field(gt=0)
    num_leaves: int = Field(gt=1)
    max_depth: int = -1
    min_data_in_leaf: int = Field(gt=0)
    feature_fraction: float = Field(gt=0, le=1)
    bagging_fraction: float = Field(gt=0, le=1)
    bagging_freq: int = Field(ge=0)
    lambda_l1: float = Field(ge=0)
    lambda_l2: float = Field(ge=0)
    num_boost_round: int = Field(gt=0)
    early_stopping_rounds: int = Field(gt=0)
    seed: int
    verbosity: int = -1

    def booster_params(self, *, num_class: int) -> dict[str, object]:
        """LightGBM ``params`` dict, minus the fields consumed separately by
        ``lgb.train`` itself (``num_boost_round``, ``early_stopping_rounds``).
        """
        return {
            "objective": self.objective,
            "num_class": num_class,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "verbosity": self.verbosity,
            "deterministic": True,
        }


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    random_baseline_seed: int
    time_buckets: int = Field(gt=1)


class TrainingConfig(BaseModel):
    """Root config: everything logged to MLflow as run params on every run."""

    model_config = ConfigDict(frozen=True)

    config_version: int
    labeling: LabelingConfig
    split: SplitConfig
    lightgbm: LightGBMConfig
    evaluation: EvaluationConfig


def load_training_config(path: Path | None = None) -> TrainingConfig:
    """Parse and validate the YAML hyperparameter file.

    Raises ``pydantic.ValidationError`` on a malformed or out-of-range value
    rather than silently coercing it -- a bad config should fail before a
    training run starts, not produce a model nobody can explain.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TrainingConfig.model_validate(raw)

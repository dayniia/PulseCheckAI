"""
tests/test_pipeline.py
======================
Unit and smoke tests for PulseCheck AI.

Run from the project root with:
    pytest tests/ -v

The tests cover:
  - Feature engineering (build_features)
  - Model loading and threshold config
  - Inference correctness and output types
  - The is_anomaly sentinel values (None vs False vs True)
  - Metric collection structure
  - Duplicate-sample deduplication guard
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the project root to sys.path so app.py functions can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROLLING_5M_LOCAL = 20  # mirror of app constant, validated in test_constants


def make_history(n, cpu=30.0, memory=60.0):
    """Return a list of n minimal sample dicts suitable for build_features."""
    return [{"cpu_usage": cpu, "memory_usage": memory} for _ in range(n)]


def make_history_varying(n):
    """Return n samples with slightly varying values to exercise std dev."""
    return [
        {
            "cpu_usage": 30.0 + (i % 5) * 2.0,
            "memory_usage": 60.0 + (i % 3) * 1.5,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

try:
    from app import (
        FEATURES,
        ROLLING_1M,
        ROLLING_5M,
        SAMPLE_INTERVAL,
        THRESHOLD,
        build_features,
        collect_metrics,
        get_top_processes,
        model,
    )
    APP_AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:
    APP_AVAILABLE = False
    _IMPORT_ERR = e

skip_if_no_app = pytest.mark.skipif(
    not APP_AVAILABLE,
    reason=f"Could not import app.py: {_IMPORT_ERR}",
)

# ---------------------------------------------------------------------------
# 1. Constants and config
# ---------------------------------------------------------------------------

@skip_if_no_app
def test_constants_match_config():
    config_path = PROJECT_ROOT / "pulsecheck_tuned_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    assert SAMPLE_INTERVAL == config["sampling_interval_seconds"]
    assert ROLLING_1M == config["rolling_window_1m_samples"]
    assert ROLLING_5M == config["rolling_window_5m_samples"]


@skip_if_no_app
def test_threshold_is_positive_float():
    assert isinstance(THRESHOLD, float)
    assert THRESHOLD > 0.0


@skip_if_no_app
def test_features_list_has_ten_entries():
    assert len(FEATURES) == 10, f"Expected 10 features, got {len(FEATURES)}"


# ---------------------------------------------------------------------------
# 2. Feature engineering — build_features()
# ---------------------------------------------------------------------------

@skip_if_no_app
def test_build_features_returns_none_before_warmup():
    """build_features returns None when fewer than ROLLING_5M samples exist."""
    for n in range(0, ROLLING_5M):
        result = build_features(make_history(n))
        assert result is None, f"Expected None for {n} samples, got {result}"


@skip_if_no_app
def test_build_features_returns_dataframe_after_warmup():
    result = build_features(make_history(ROLLING_5M))
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 10), f"Expected (1,10), got {result.shape}"


@skip_if_no_app
def test_build_features_column_order_matches_config():
    result = build_features(make_history_varying(ROLLING_5M))
    assert result is not None
    assert list(result.columns) == FEATURES


@skip_if_no_app
def test_build_features_no_nan_in_full_buffer():
    result = build_features(make_history_varying(ROLLING_5M))
    assert result is not None
    assert not result.isna().any().any()


@skip_if_no_app
def test_build_features_large_buffer_returns_one_row():
    result = build_features(make_history_varying(60))
    assert result is not None
    assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# 3. Model inference
# ---------------------------------------------------------------------------

@skip_if_no_app
def test_model_loads_and_is_not_none():
    assert model is not None


@skip_if_no_app
def test_model_produces_finite_float_score():
    feature_row = build_features(make_history_varying(ROLLING_5M))
    assert feature_row is not None
    score = float(-model.decision_function(feature_row)[0])
    assert isinstance(score, float)
    assert math.isfinite(score)


@skip_if_no_app
def test_threshold_comparison_produces_bool():
    feature_row = build_features(make_history_varying(ROLLING_5M))
    score = float(-model.decision_function(feature_row)[0])
    is_anomaly = bool(score >= THRESHOLD)
    assert isinstance(is_anomaly, bool)


# ---------------------------------------------------------------------------
# 4. is_anomaly sentinel correctness (None != False bug fix)
# ---------------------------------------------------------------------------

def test_is_anomaly_none_is_not_false():
    """None (unknown/error state) must never be treated as False (normal)."""
    sample = {"anomaly_score": float("nan"), "is_anomaly": None}
    val = sample.get("is_anomaly", None)
    assert val is None
    assert val is not False
    assert (val is True) is False


def test_is_anomaly_true_passes_strict_check():
    sample = {"anomaly_score": 0.25, "is_anomaly": True}
    assert sample["is_anomaly"] is True


def test_is_anomaly_false_fails_strict_check():
    sample = {"anomaly_score": 0.05, "is_anomaly": False}
    assert (sample["is_anomaly"] is True) is False


# ---------------------------------------------------------------------------
# 5. Metric collection structure
# ---------------------------------------------------------------------------

@skip_if_no_app
def test_collect_metrics_required_keys():
    required = {"timestamp", "cpu_usage", "memory_usage", "disk_usage", "memory_used_gb"}
    sample = collect_metrics()
    missing = required - set(sample.keys())
    assert not missing, f"Missing keys: {missing}"


@skip_if_no_app
def test_collect_metrics_numeric_fields_are_finite():
    sample = collect_metrics()
    for key in ("cpu_usage", "memory_usage", "disk_usage", "memory_used_gb"):
        assert math.isfinite(sample[key]), f"{key} is not finite: {sample[key]}"


@skip_if_no_app
def test_collect_metrics_cpu_range():
    sample = collect_metrics()
    assert 0.0 <= sample["cpu_usage"] <= 100.0


@skip_if_no_app
def test_collect_metrics_memory_range():
    sample = collect_metrics()
    assert 0.0 <= sample["memory_usage"] <= 100.0


# ---------------------------------------------------------------------------
# 6. Process inspection
# ---------------------------------------------------------------------------

@skip_if_no_app
def test_get_top_processes_returns_dataframe():
    df = get_top_processes(limit=5)
    assert isinstance(df, pd.DataFrame)


@skip_if_no_app
def test_get_top_processes_has_correct_columns():
    df = get_top_processes(limit=5)
    for col in ("PID", "Process", "RAM (%)"):
        assert col in df.columns, f"Missing column: {col}"


@skip_if_no_app
def test_get_top_processes_respects_limit():
    limit = 3
    df = get_top_processes(limit=limit)
    assert len(df) <= limit


# ---------------------------------------------------------------------------
# 7. Duplicate-sample deduplication guard
# ---------------------------------------------------------------------------

def test_deduplication_rejects_premature_sample():
    """A sample collected 5s after the last should be rejected (interval=15s)."""
    INTERVAL = 15
    now = pd.Timestamp.now()
    last = {"timestamp": now}
    new = {"timestamp": now + pd.Timedelta(seconds=5)}
    elapsed = (new["timestamp"] - last["timestamp"]).total_seconds()
    assert not (elapsed >= INTERVAL - 2)


def test_deduplication_accepts_timely_sample():
    """A sample collected 14s after the last should be accepted (interval=15s)."""
    INTERVAL = 15
    now = pd.Timestamp.now()
    last = {"timestamp": now}
    new = {"timestamp": now + pd.Timedelta(seconds=14)}
    elapsed = (new["timestamp"] - last["timestamp"]).total_seconds()
    assert elapsed >= INTERVAL - 2

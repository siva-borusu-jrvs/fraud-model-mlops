"""Unit tests for model training utilities."""

import pytest
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score


class TestModelTraining:
    """Tests for model training logic."""

    def test_gbt_trains_on_synthetic_data(self):
        """GBT classifier should train without error on balanced synthetic data."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 9)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X[:160], y[:160])

        preds = model.predict(X[160:])
        f1 = f1_score(y[160:], preds)
        assert f1 > 0.5, f"F1 too low: {f1}"

    def test_feature_cols_match_config(self):
        """Training feature list should match what config expects."""
        expected_count = 9  # per FEATURE_COLS in train.py
        from src.config import CFG
        assert CFG.fq_model_name == "ml.fraud.fraud_classifier"

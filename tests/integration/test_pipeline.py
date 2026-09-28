"""Integration tests for the end-to-end pipeline.

Run these on a Databricks cluster with access to Unity Catalog.
"""

import pytest
from src.config import CFG


@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end pipeline validation (requires cluster + UC access)."""

    def test_raw_table_exists(self, spark):
        """Raw transactions table should be accessible."""
        assert spark.catalog.tableExists(CFG.fq_raw_table)

    def test_feature_table_exists(self, spark):
        """Feature table should be accessible."""
        assert spark.catalog.tableExists(CFG.fq_feature_table)

    def test_feature_table_has_expected_columns(self, spark):
        """Feature table should contain all engineered columns."""
        df = spark.table(CFG.fq_feature_table)
        expected = {"txn_count_1d", "txn_count_7d", "amount_zscore", "hour_of_day"}
        actual = set(df.columns)
        assert expected.issubset(actual), f"Missing columns: {expected - actual}"

    def test_model_registered(self):
        """Champion model should be registered in Unity Catalog."""
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.search_model_versions(f"name=\'{CFG.fq_model_name}\'")
        assert len(versions) > 0, "No model versions found"

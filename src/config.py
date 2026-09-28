"""Central configuration for the fraud detection project.

All catalog paths, model names, endpoint names, and feature table
references live here so every module pulls from a single source of truth.

Environment-aware: when run via a DABs job, reads catalog/schema/model_name/
endpoint_name from dbutils.widgets (passed as job base_parameters).
When run interactively, falls back to dev defaults.
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Resolve bundle variables from job parameters ───────────────────
def _get_param(key: str, default: str) -> str:
    """Read a job parameter (widget) if available, else return default."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark:
            return spark.conf.get(f"spark.databricks.workflow.parameters.{key}", default)
    except Exception:
        pass
    try:
        # Fallback: try dbutils widgets directly
        return dbutils.widgets.get(key)  # noqa: F821
    except Exception:
        return default


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable project-wide settings.

    Defaults are dev values. When deployed via DABs, job parameters
    override catalog, schema, model_name, and endpoint_name.
    """

    # ── Unity Catalog coordinates (overridable via DABs) ───────────
    catalog: str = "ml_dev"
    schema: str = "fraud_dev"

    # ── Tables ─────────────────────────────────────────────────────
    raw_transactions_table: str = "raw_transactions"
    feature_table: str = "fraud_features"
    predictions_table: str = "fraud_predictions"
    drift_metrics_table: str = "drift_metrics"

    # ── Model (overridable via DABs) ───────────────────────────────
    registered_model_name: str = "fraud_classifier_dev"
    experiment_name: str = "/Shared/fraud-model-experiment-dev"

    # ── Serving (overridable via DABs) ─────────────────────────────
    endpoint_name: str = "fraud-classifier-dev"
    endpoint_workload_size: str = "Small"
    endpoint_scale_to_zero: bool = True

    # ── Monitoring / Alerts ────────────────────────────────────────
    drift_threshold_psi: float = 0.2
    drift_threshold_ks: float = 0.05
    alert_email_recipients: list = field(
        default_factory=lambda: ["siva.borusu@jrvs.ca"]
    )

    # ── Cluster ────────────────────────────────────────────────────
    cluster_policy_name: Optional[str] = None  # None -> use serverless

    # ── Helpers ────────────────────────────────────────────────────
    @property
    def fq_raw_table(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.raw_transactions_table}"

    @property
    def fq_feature_table(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.feature_table}"

    @property
    def fq_predictions_table(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.predictions_table}"

    @property
    def fq_drift_table(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.drift_metrics_table}"

    @property
    def fq_model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.registered_model_name}"


def build_config() -> ProjectConfig:
    """Build config from job parameters (DABs) or use dev defaults."""
    catalog = _get_param("catalog", "ml_dev")
    schema = _get_param("schema", "fraud_dev")
    model_name = _get_param("model_name", "fraud_classifier_dev")
    endpoint_name = _get_param("endpoint_name", "fraud-classifier-dev")

    return ProjectConfig(
        catalog=catalog,
        schema=schema,
        registered_model_name=model_name,
        experiment_name=f"/Shared/fraud-model-experiment-{catalog.split('_')[-1] if '_' in catalog else 'prod'}",
        endpoint_name=endpoint_name,
    )


# Singleton - import this in other modules.
# Resolves env at import time from job params or defaults to dev.
CFG = build_config()

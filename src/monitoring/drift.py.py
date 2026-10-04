# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Lakehouse Drift Monitoring
# MAGIC Uses Databricks Quality Monitoring to track feature, prediction, and model quality drift.
# MAGIC Monitors `customer_n_transactions_drift` (scored through the champion model) against
# MAGIC the `customer_n_transactions` baseline. Includes both `is_fraud` labels and `prediction`
# MAGIC columns, enabling data drift, prediction drift, and concept drift detection.
# MAGIC Outputs profile and drift metrics to `{catalog}.{schema}`.
# MAGIC Parameterized for multi-environment deployment via DABs job base_parameters.

# COMMAND ----------

# DBTITLE 1,Imports and Configuration
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorCronSchedule,
)
import mlflow
import pyspark.sql.functions as F
from pyspark.sql.functions import struct
from src.utils import get_logger

logger = get_logger("monitoring.drift")

# ── Resolve parameters from DABs job or fall back to dev defaults ──
def _get_param(key: str, default: str) -> str:
    try:
        return spark.conf.get(f"spark.databricks.workflow.parameters.{key}", default)
    except Exception:
        pass
    try:
        return dbutils.widgets.get(key)
    except Exception:
        return default

_catalog = _get_param("catalog", "mlops")
_schema = _get_param("schema", "siva_borusu")
_model_name = _get_param("model_name", "fraud_model_credit_card")

# ── Drift monitoring configuration ─────────────────────────────────
DRIFT_CONFIG = {
    # Tables — built from job parameters for multi-environment support
    "base_table": f"{_catalog}.{_schema}.customer_n_transactions_baseline",
    "drift_table": f"{_catalog}.{_schema}.customer_n_transactions_drift",
    "output_schema": f"{_catalog}.{_schema}",
    # Model used to generate predictions for drift tracking
    "model_uri": f"models:/{_catalog}.{_schema}.{_model_name}@champion",
    # Column mapping
    "prediction_col": "prediction",
    "label_col": "is_fraud",
    "timestamp_col": "transaction_timestamp",
    "model_id_col": "model_version",
    # Monitor settings
    "problem_type": MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
    "granularities": ["1 day", "1 week"],
    # Refresh schedule — daily at 08:00 America/Toronto
    "schedule_cron": "0 0 8 * * ?",
    "schedule_timezone": "America/Toronto",
    # Columns to exclude from scoring (IDs + label)
    "exclude_cols": ["transaction_id", "card_id", "is_fraud"],
    # Slicing expressions — compute drift metrics per slice
    "slicing_exprs": ["home_country", "home_state"],
    # Workspace path for auto-generated dashboard assets (shared, not user-specific)
    "assets_dir": "/Workspace/Shared/fraud-model-mlops/monitoring_assets",
}

# COMMAND ----------

# DBTITLE 1,DriftMonitor Class
# MAGIC %md
# MAGIC ## DriftMonitor Class

# COMMAND ----------

# DBTITLE 1,DriftMonitor Class Definition
class DriftMonitor:
    """Manages Databricks Lakehouse Monitoring for fraud model drift detection.

    Workflow
    --------
    1. score_drift_table()    – batch-score the drift table through the champion model
    2. create_drift_monitor() – register a Quality Monitor (InferenceLog type)
    3. refresh_monitor()      – trigger an on-demand metrics refresh
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.w = WorkspaceClient()

    # ── Score drift table with the champion model ─────────────────
    def score_drift_table(self) -> None:
        """Add prediction, model_version, and scored_at columns to the drift table.

        Uses mlflow.pyfunc.spark_udf for distributed scoring.  The model's
        input signature determines which columns are consumed as features.
        """
        predict_udf = mlflow.pyfunc.spark_udf(
            spark, model_uri=self.cfg["model_uri"], result_type="double"
        )

        df = spark.table(self.cfg["drift_table"])

        # Exclude metadata columns that are not model features
        meta_cols = {
            self.cfg["prediction_col"],
            self.cfg["timestamp_col"],
            self.cfg["model_id_col"],
        }
        meta_cols.update(self.cfg.get("exclude_cols", []))
        feature_cols = [c for c in df.columns if c not in meta_cols]

        scored_df = (
            df.select(*feature_cols)
            .withColumn(self.cfg["prediction_col"], predict_udf(struct(*feature_cols)))
            .withColumn(self.cfg["timestamp_col"], F.current_timestamp())
            .withColumn(self.cfg["model_id_col"], F.lit(self.cfg["model_uri"]))
        )

        scored_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(self.cfg["drift_table"])

        logger.info(f"Scored {scored_df.count()} rows -> {self.cfg['drift_table']}")

    # ── Create the Lakehouse Quality Monitor ──────────────────────
    def create_drift_monitor(self) -> object:
        """Register a Quality Monitor (InferenceLog) on the drift table.

        Compares against the baseline (customer_features) for data drift
        and tracks prediction distribution over time for model drift.
        Returns the MonitorInfo object.
        """
        inference_log = MonitorInferenceLog(
            granularities=self.cfg["granularities"],
            model_id_col=self.cfg["model_id_col"],
            prediction_col=self.cfg["prediction_col"],
            label_col=self.cfg.get("label_col"),
            timestamp_col=self.cfg["timestamp_col"],
            problem_type=self.cfg["problem_type"],
        )

        schedule = MonitorCronSchedule(
            quartz_cron_expression=self.cfg["schedule_cron"],
            timezone_id=self.cfg["schedule_timezone"],
        )

        monitor = self.w.quality_monitors.create(
            table_name=self.cfg["drift_table"],
            assets_dir=self.cfg["assets_dir"],
            output_schema_name=self.cfg["output_schema"],
            baseline_table_name=self.cfg["base_table"],
            inference_log=inference_log,
            schedule=schedule,
            slicing_exprs=self.cfg.get("slicing_exprs"),
        )

        tbl = self.cfg["drift_table"].split(".")[-1]
        schema = self.cfg["output_schema"]
        logger.info(f"Monitor created on {self.cfg['drift_table']}")
        logger.info(f"  Profile table -> {schema}.{tbl}_profile_metrics")
        logger.info(f"  Drift table   -> {schema}.{tbl}_drift_metrics")
        logger.info(f"  Schedule      -> {self.cfg['schedule_cron']} ({self.cfg['schedule_timezone']})")
        return monitor

    # ── Helper methods ────────────────────────────────────────────
    def get_monitor(self) -> object:
        """Retrieve current monitor status."""
        return self.w.quality_monitors.get(table_name=self.cfg["drift_table"])

    def refresh_monitor(self) -> object:
        """Trigger an on-demand refresh of the monitor metrics."""
        refresh = self.w.quality_monitors.run_refresh(
            table_name=self.cfg["drift_table"]
        )
        logger.info(f"Refresh triggered: {refresh.refresh_id}")
        return refresh

    def delete_monitor(self) -> None:
        """Remove the monitor (does not delete output tables)."""
        self.w.quality_monitors.delete(table_name=self.cfg["drift_table"])
        logger.info(f"Monitor deleted from {self.cfg['drift_table']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# DBTITLE 1,Run
dm = DriftMonitor(DRIFT_CONFIG)
monitor = dm.create_drift_monitor()

# Wait for monitor to leave PENDING, then trigger first refresh
import time
for _ in range(6):
    status = dm.get_monitor()
    if hasattr(status, 'status') and 'PENDING' not in str(status.status):
        break
    logger.info("Monitor initializing, waiting 10s...")
    time.sleep(10)

try:
    dm.refresh_monitor()
except Exception as e:
    logger.warning(f"Refresh skipped (monitor may still be initializing): {e}")
    logger.info("The scheduled cron will pick it up, or re-run dm.refresh_monitor() later.")
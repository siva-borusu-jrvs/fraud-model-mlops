# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # A/B Test Resolution
# MAGIC Resolves the champion/challenger A/B test on the serving endpoint.
# MAGIC Checks production monitoring metrics, archives the loser with a timestamped alias,
# MAGIC promotes the winner to `@champion`, and drives 100% traffic to it.
# MAGIC
# MAGIC Runs on a **separate schedule** from the retrain pipeline (e.g., daily),
# MAGIC after enough production traffic has been collected on the 50/50 split.

# COMMAND ----------

# DBTITLE 1,Imports and Config
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, TrafficConfig, Route
from mlflow.tracking import MlflowClient
from src.config import CFG
from src.utils import get_logger

logger = get_logger("serving.ab_test")
w = WorkspaceClient()
mlflow_client = MlflowClient(registry_uri="databricks-uc")

_pyfunc_model = f"{CFG.fq_model_name}_pyfunc"
_profile_table = f"{CFG.catalog}.{CFG.schema}.customer_n_transactions_drift_profile_metrics"
_drift_table = f"{CFG.catalog}.{CFG.schema}.customer_n_transactions_drift_drift_metrics"

logger.info(f"Pyfunc model:  {_pyfunc_model}")
logger.info(f"Endpoint:      {CFG.endpoint_name}")
logger.info(f"Profile table: {_profile_table}")

# COMMAND ----------

# DBTITLE 1,Check Active Split
# MAGIC %md
# MAGIC ## Check Active A/B Split

# COMMAND ----------

# DBTITLE 1,Check Active Split
def check_active_split():
    """Verify that both @champion and @challenger exist on the pyfunc model.

    Returns (champion_mv, challenger_mv) or raises if no active split.
    """
    try:
        champion_mv = mlflow_client.get_model_version_by_alias(_pyfunc_model, "champion")
    except Exception:
        raise RuntimeError("No @champion alias on pyfunc model. Nothing to resolve.")

    try:
        challenger_mv = mlflow_client.get_model_version_by_alias(_pyfunc_model, "challenger")
    except Exception:
        logger.info("No @challenger alias found — no active A/B test to resolve.")
        return None, None

    if champion_mv.version == challenger_mv.version:
        logger.info(f"Champion and challenger are the same version (v{champion_mv.version}) — nothing to resolve.")
        return None, None

    logger.info(f"Active A/B split: @champion v{champion_mv.version} vs @challenger v{challenger_mv.version}")
    return champion_mv, challenger_mv

# COMMAND ----------

# DBTITLE 1,Get Metrics
# MAGIC %md
# MAGIC ## Get Metrics
# MAGIC Use the **1 week** granularity from monitoring profile_metrics (matches the
# MAGIC `["1 day", "1 week"]` granularities configured in `drift.py`).
# MAGIC Fall back to MLflow training metrics if monitoring data is unavailable.

# COMMAND ----------

# DBTITLE 1,Get Metrics
def _get_mlflow_metric(version: str, metric: str) -> float:
    """Get a training metric from the MLflow run for a pyfunc model version."""
    mv = mlflow_client.get_model_version(_pyfunc_model, str(version))
    return mlflow_client.get_run(mv.run_id).data.metrics.get(metric, 0.0)


def get_model_scores(champion_mv, challenger_mv, primary_metric="recall"):
    """Retrieve per-model scores from monitoring or MLflow.

    Priority:
    1. Lakehouse Monitor profile_metrics (production quality per model_id)
    2. MLflow training metrics (offline, logged by pyfunc_wrapper.py)

    Returns (champ_score, chal_score, source_label).
    """
    # --- Try monitoring profile_metrics (1 week granularity) ---
    try:
        df = spark.sql(f"""
            SELECT model_id,
                   {primary_metric} AS score,
                   window.end AS window_end
            FROM   {_profile_table}
            WHERE  granularity = '1 week'
              AND  column_name = ':table'
              AND  slice_key IS NULL
              AND  {primary_metric} IS NOT NULL
            ORDER BY window.end DESC
        """)
        # Take the most recent 1-week window per model_id
        rows = {}
        for r in df.collect():
            if r["model_id"] not in rows:
                rows[r["model_id"]] = r["score"]
        if len(rows) >= 2:
            # model_id in the monitor is the model_uri string
            champ_id = f"models:/{_pyfunc_model}@champion"
            chal_id = f"models:/{_pyfunc_model}@challenger"
            if champ_id in rows and chal_id in rows:
                logger.info("Using production monitoring metrics (1 week granularity)")
                return rows[champ_id], rows[chal_id], "monitoring_1week"
    except Exception as e:
        logger.warning(f"Monitoring metrics unavailable: {e}")

    # --- Fallback: MLflow training metrics ---
    logger.info("Falling back to MLflow training metrics")
    champ_score = _get_mlflow_metric(champion_mv.version, primary_metric)
    chal_score = _get_mlflow_metric(challenger_mv.version, primary_metric)
    return champ_score, chal_score, "mlflow_training"

# COMMAND ----------

# DBTITLE 1,Resolve A/B Test
# MAGIC %md
# MAGIC ## Resolve A/B Test

# COMMAND ----------

# DBTITLE 1,Resolve A/B Test
def resolve_ab_test(primary_metric: str = "recall"):
    """Compare champion vs challenger and resolve the A/B test.

    1. Fetch scores from monitoring (or MLflow fallback)
    2. Archive the loser with alias archive_YYYYMMDD_HHMMSS
    3. Promote the winner to @champion
    4. Remove @challenger alias
    5. Update endpoint to 100% traffic to winner
    """
    champion_mv, challenger_mv = check_active_split()
    if champion_mv is None:
        return None  # No active split

    # --- Score comparison ---
    champ_score, chal_score, source = get_model_scores(
        champion_mv, challenger_mv, primary_metric
    )
    logger.info(f"@champion  (v{champion_mv.version}): {primary_metric} = {champ_score:.4f}")
    logger.info(f"@challenger (v{challenger_mv.version}): {primary_metric} = {chal_score:.4f}")
    logger.info(f"Source: {source}")

    # --- Determine winner/loser ---
    if chal_score > champ_score:
        winner_mv, loser_mv = challenger_mv, champion_mv
        winner_label = "challenger"
    else:
        winner_mv, loser_mv = champion_mv, challenger_mv
        winner_label = "champion"

    logger.info(f"Winner: {winner_label} (v{winner_mv.version})")

    # --- Archive the loser ---
    archive_alias = f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_client.set_registered_model_alias(_pyfunc_model, archive_alias, loser_mv.version)
    logger.info(f"Archived v{loser_mv.version} as @{archive_alias}")

    # --- Promote winner to @champion ---
    mlflow_client.set_registered_model_alias(_pyfunc_model, "champion", winner_mv.version)
    logger.info(f"Promoted v{winner_mv.version} to @champion")

    # --- Remove @challenger alias ---
    try:
        mlflow_client.delete_registered_model_alias(_pyfunc_model, "challenger")
        logger.info("Removed @challenger alias")
    except Exception:
        pass

    # --- Update endpoint: 100% traffic to winner ---
    winner_entity = ServedEntityInput(
        entity_name=_pyfunc_model,
        entity_version=winner_mv.version,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
        name="champion",
    )
    traffic = TrafficConfig(routes=[
        Route(served_model_name="champion", traffic_percentage=100),
    ])
    w.serving_endpoints.update_config(
        name=CFG.endpoint_name,
        served_entities=[winner_entity],
        traffic_config=traffic,
    )
    logger.info(f"Endpoint {CFG.endpoint_name}: 100% traffic to v{winner_mv.version}")

    return {
        "winner_version": winner_mv.version,
        "loser_version": loser_mv.version,
        "winner_metric": max(champ_score, chal_score),
        "loser_metric": min(champ_score, chal_score),
        "archive_alias": archive_alias,
        "metric_source": source,
    }

# COMMAND ----------

# DBTITLE 1,Run
# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# DBTITLE 1,Run
result = resolve_ab_test(primary_metric="recall")
if result:
    logger.info(f"Resolution complete: {result}")
else:
    logger.info("No active A/B test to resolve.")
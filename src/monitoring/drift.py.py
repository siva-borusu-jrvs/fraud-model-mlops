# Databricks notebook source
# MAGIC %md
# MAGIC # Model Drift Monitoring
# MAGIC Computes PSI between training and recent prediction distributions.
# MAGIC Writes results to a drift metrics Delta table for SQL alert triggers.

# COMMAND ----------

import numpy as np
import pyspark.sql.functions as F
from datetime import datetime, timedelta
from src.config import CFG

# COMMAND ----------

# MAGIC %md
# MAGIC ## PSI Calculation

# COMMAND ----------

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pcts = np.clip(expected_pcts, 1e-6, None)
    actual_pcts = np.clip(actual_pcts, 1e-6, None)
    return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Drift Metrics

# COMMAND ----------

MONITORED_FEATURES = [
    "txn_count_1d", "txn_count_7d", "txn_amount_avg_1d",
    "txn_amount_avg_7d", "amount_zscore", "amount",
]

def compute_drift_metrics(lookback_days: int = 7):
    """Compare training distributions against recent predictions."""
    training_df = spark.table(CFG.fq_feature_table).toPandas()
    cutoff = datetime.now() - timedelta(days=lookback_days)
    predictions_df = (
        spark.table(CFG.fq_predictions_table)
        .filter(F.col("prediction_ts") >= cutoff)
        .toPandas()
    )
    if predictions_df.empty:
        print("No recent predictions - skipping drift check.")
        return None

    results = {"run_date": datetime.now().isoformat(), "lookback_days": lookback_days}
    for col in MONITORED_FEATURES:
        if col in training_df.columns and col in predictions_df.columns:
            psi = compute_psi(training_df[col].dropna().values, predictions_df[col].dropna().values)
            results[f"{col}_psi"] = psi
            flag = " ⚠ DRIFT" if psi > CFG.drift_threshold_psi else ""
            print(f"  {col}: PSI={psi:.4f}{flag}")
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Drift Table

# COMMAND ----------

def write_drift_metrics(metrics: dict):
    """Append drift metrics to the monitoring Delta table."""
    df = spark.createDataFrame([metrics])
    df.write.format("delta").mode("append").saveAsTable(CFG.fq_drift_table)
    print(f"Written to {CFG.fq_drift_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# metrics = compute_drift_metrics(lookback_days=7)
# if metrics:
#     write_drift_metrics(metrics)
#     display(spark.table(CFG.fq_drift_table).orderBy(F.desc("run_date")).limit(10))

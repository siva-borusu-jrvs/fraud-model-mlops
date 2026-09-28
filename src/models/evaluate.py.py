# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation
# MAGIC Compares candidate vs champion model; promotes if it beats the primary metric.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from src.config import CFG

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_latest_version(alias: str = "champion"):
    """Return model version number for the given alias, or None."""
    try:
        mv = client.get_model_version_by_alias(CFG.fq_model_name, alias)
        return int(mv.version)
    except mlflow.exceptions.MlflowException:
        return None


def _set_alias(run_id: str, alias: str):
    """Set alias on the model version produced by the given run."""
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if versions:
        client.set_registered_model_alias(CFG.fq_model_name, alias, versions[0].version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare & Promote

# COMMAND ----------

def compare_and_promote(candidate_run_id: str, primary_metric: str = "roc_auc", min_improvement: float = 0.005) -> bool:
    """Compare candidate to champion; promote if better."""
    champion_version = get_latest_version("champion")
    if champion_version is None:
        _set_alias(candidate_run_id, "champion")
        print("No champion yet - auto-promoted candidate.")
        return True

    champion_run = client.get_model_version(CFG.fq_model_name, champion_version)
    champ_score = client.get_run(champion_run.run_id).data.metrics.get(primary_metric, 0)
    cand_score = client.get_run(candidate_run_id).data.metrics.get(primary_metric, 0)

    print(f"Champion {primary_metric}: {champ_score:.4f}")
    print(f"Candidate {primary_metric}: {cand_score:.4f}")

    if cand_score >= champ_score + min_improvement:
        _set_alias(candidate_run_id, "champion")
        client.set_registered_model_alias(CFG.fq_model_name, "previous-champion", champion_version)
        print("Candidate PROMOTED to champion.")
        return True
    print("Champion retained.")
    return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# candidate_run_id = "<run-id-from-training>"
# compare_and_promote(candidate_run_id)

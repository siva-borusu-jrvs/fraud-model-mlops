# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Model Deployment
# MAGIC Deploys champion and challenger pyfunc models to the serving endpoint with 50/50 traffic split.
# MAGIC Pure deployment — no alias management. Assumes the endpoint already exists.

# COMMAND ----------

# DBTITLE 1,Imports and clients
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, TrafficConfig, Route
from mlflow.tracking import MlflowClient
from src.config import CFG
from src.utils import get_logger

logger = get_logger("serving.deploy")
w = WorkspaceClient()
mlflow_client = MlflowClient(registry_uri="databricks-uc")

_pyfunc_model = f"{CFG.fq_model_name}_pyfunc"
logger.info(f"Pyfunc model: {_pyfunc_model}")
logger.info(f"Endpoint:     {CFG.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Endpoint

# COMMAND ----------

# DBTITLE 1,Deploy champion and challenger
def deploy():
    """Deploy champion and challenger to the endpoint with 50/50 traffic split.

    Pure deployment — resolves @champion and @challenger aliases on the pyfunc
    model and updates the existing endpoint. No alias management happens here.
    """
    # Resolve aliases to version numbers
    champion_mv = mlflow_client.get_model_version_by_alias(_pyfunc_model, "champion")
    challenger_mv = mlflow_client.get_model_version_by_alias(_pyfunc_model, "challenger")

    logger.info(f"@champion  -> v{champion_mv.version}")
    logger.info(f"@challenger -> v{challenger_mv.version}")

    # Env vars (DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_SQL_WAREHOUSE_ID)
    # are already configured at the endpoint level — no need to pass here.
    champion_entity = ServedEntityInput(
        entity_name=_pyfunc_model,
        entity_version=champion_mv.version,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
        name="champion",
    )
    challenger_entity = ServedEntityInput(
        entity_name=_pyfunc_model,
        entity_version=challenger_mv.version,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
        name="challenger",
    )

    traffic = TrafficConfig(routes=[
        Route(served_model_name="champion", traffic_percentage=50),
        Route(served_model_name="challenger", traffic_percentage=50),
    ])

    logger.info(f"Updating endpoint: {CFG.endpoint_name} (50/50 traffic split)")
    w.serving_endpoints.update_config(
        name=CFG.endpoint_name,
        served_entities=[champion_entity, challenger_entity],
        traffic_config=traffic,
    )
    logger.info("Deployment initiated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# DBTITLE 1,Run
deploy()
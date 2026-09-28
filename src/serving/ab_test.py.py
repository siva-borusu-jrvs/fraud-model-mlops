# Databricks notebook source
# MAGIC %md
# MAGIC # A/B Testing Setup
# MAGIC Configures traffic routing between champion and challenger on the serving endpoint.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route
from src.config import CFG

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Traffic Split

# COMMAND ----------

def setup_ab_test(champion_version: int, challenger_version: int, challenger_pct: int = 10):
    """Route traffic between champion and challenger versions."""
    champion_entity = ServedEntityInput(
        entity_name=CFG.fq_model_name, entity_version=str(champion_version),
        workload_size=CFG.endpoint_workload_size, scale_to_zero_enabled=CFG.endpoint_scale_to_zero, name="champion",
    )
    challenger_entity = ServedEntityInput(
        entity_name=CFG.fq_model_name, entity_version=str(challenger_version),
        workload_size=CFG.endpoint_workload_size, scale_to_zero_enabled=CFG.endpoint_scale_to_zero, name="challenger",
    )
    traffic = TrafficConfig(routes=[
        Route(served_model_name="champion", traffic_percentage=100 - challenger_pct),
        Route(served_model_name="challenger", traffic_percentage=challenger_pct),
    ])
    w.serving_endpoints.update_config(
        name=CFG.endpoint_name,
        served_entities=[champion_entity, challenger_entity],
        traffic_config=traffic,
    )
    print(f"A/B test: champion={100 - challenger_pct}%, challenger={challenger_pct}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Table Logging
# MAGIC Predictions auto-logged to: `{catalog}.{schema}.{endpoint}_payload`

# COMMAND ----------

def get_inference_table_name():
    """Return the inference table name for the endpoint."""
    table = f"{CFG.catalog}.{CFG.schema}.{CFG.endpoint_name.replace('-', '_')}_payload"
    print(f"Inference table: {table}")
    return table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# setup_ab_test(champion_version=1, challenger_version=2, challenger_pct=10)
# get_inference_table_name()

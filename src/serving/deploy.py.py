# Databricks notebook source
# MAGIC %md
# MAGIC # Model Deployment
# MAGIC Deploys the champion model version to a Model Serving endpoint.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from src.config import CFG

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Endpoint

# COMMAND ----------

def deploy_champion():
    """Deploy the champion alias to the serving endpoint."""
    served_entity = ServedEntityInput(
        entity_name=CFG.fq_model_name,
        entity_version=None,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
    )
    config = EndpointCoreConfigInput(served_entities=[served_entity])

    existing = [e.name for e in w.serving_endpoints.list()]
    if CFG.endpoint_name in existing:
        print(f"Updating endpoint: {CFG.endpoint_name}")
        w.serving_endpoints.update_config(name=CFG.endpoint_name, served_entities=[served_entity])
    else:
        print(f"Creating endpoint: {CFG.endpoint_name}")
        w.serving_endpoints.create(name=CFG.endpoint_name, config=config)
    print("Deployment initiated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Endpoint Status

# COMMAND ----------

def check_endpoint_status():
    """Print the current state of the serving endpoint."""
    ep = w.serving_endpoints.get(CFG.endpoint_name)
    print(f"Endpoint: {ep.name}  |  State: {ep.state.ready}")
    for entity in (ep.config.served_entities or []):
        print(f"  Entity: {entity.entity_name}  Version: {entity.entity_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# deploy_champion()
# check_endpoint_status()

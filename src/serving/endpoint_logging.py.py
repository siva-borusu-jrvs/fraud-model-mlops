# Databricks notebook source
# DBTITLE 1,Endpoint Logging Configuration
# MAGIC %md
# MAGIC # Endpoint Logging Configuration
# MAGIC Enables AI Gateway inference table logging and usage tracking on **any** Model Serving endpoint
# MAGIC in this project, not just the champion endpoint.
# MAGIC
# MAGIC `deploy.py` intentionally passes only `served_entities`, so it never touches AI Gateway — this
# MAGIC notebook is the single place logging gets turned on. `put_ai_gateway` is idempotent and
# MAGIC independent of model versions, so it can be re-run any time without redeploying a model.
# MAGIC
# MAGIC The inference table lands as `<catalog>.<schema>.<model_name>_payload`, a view over a
# MAGIC `_otel_logs` telemetry table. Phase 6 (Lakehouse Monitoring + drift dashboards) reads this view,
# MAGIC so every endpoint whose traffic should be monitored must appear in `ENDPOINTS` below.

# COMMAND ----------

# DBTITLE 1,Imports and clients
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
)
from src.config import CFG
from src.utils import get_logger

logger = get_logger("serving.ai_gateway")
w = WorkspaceClient()

# COMMAND ----------

# DBTITLE 1,Configure AI Gateway
def configure_ai_gateway(
    endpoint_name: str = None,
    inference_table: bool = True,
    usage_tracking: bool = True,
    catalog: str = None,
    schema: str = None,
):
    """Configure AI Gateway logging on a serving endpoint.

    Args:
        endpoint_name:   Endpoint to configure. Defaults to CFG.endpoint_name.
        inference_table: Enable inference table logging (request/response payloads).
        usage_tracking:  Enable usage tracking (request count, latency, requester).
        catalog:         Catalog for the payload table. Defaults to CFG.catalog.
        schema:          Schema for the payload table. Defaults to CFG.schema.
    """
    endpoint_name = endpoint_name or CFG.endpoint_name
    catalog = catalog or CFG.catalog
    schema = schema or CFG.schema

    w.serving_endpoints.put_ai_gateway(
        name=endpoint_name,
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=catalog,
            schema_name=schema,
            enabled=inference_table,
        ),
        usage_tracking_config=AiGatewayUsageTrackingConfig(
            enabled=usage_tracking,
        ),
    )
    logger.info(f"AI Gateway config updated on '{endpoint_name}'")
    logger.info(f"  Inference table: {'enabled' if inference_table else 'disabled'} -> {catalog}.{schema}.<model_name>_payload")
    logger.info(f"  Usage tracking:  {'enabled' if usage_tracking else 'disabled'}")

# COMMAND ----------

# DBTITLE 1,Audit logging status
def logging_status() -> pd.DataFrame:
    """Audit every serving endpoint and report its current AI Gateway logging state."""
    rows = []
    for ep in w.serving_endpoints.list():
        gw = w.serving_endpoints.get(ep.name).ai_gateway
        it = gw.inference_table_config if gw else None
        ut = gw.usage_tracking_config if gw else None
        rows.append(
            {
                "endpoint": ep.name,
                "ready": ep.state.ready.value if ep.state and ep.state.ready else None,
                "inference_table": bool(it and it.enabled),
                "payload_location": f"{it.catalog_name}.{it.schema_name}" if it and it.enabled else None,
                "usage_tracking": bool(ut and ut.enabled),
            }
        )
    return pd.DataFrame(rows)


display(logging_status())

# COMMAND ----------

# DBTITLE 1,Run
# Endpoints that should log request/response payloads for monitoring.
# Add any additional endpoint (e.g. a pyfunc or challenger endpoint) here.
ENDPOINTS = [
    CFG.endpoint_name,
]

for name in ENDPOINTS:
    configure_ai_gateway(endpoint_name=name)

display(logging_status())
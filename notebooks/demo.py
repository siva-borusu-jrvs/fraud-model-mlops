# Databricks notebook source
# DBTITLE 1,Demo Setup
# End-to-End Demo: Fraud Model MLOps Pipeline
# This notebook walks through the full workflow:
# 1. Ingest data
# 2. Build features
# 3. Train model
# 4. Evaluate & promote
# 5. Deploy to serving endpoint

from src.config import CFG
print(f"Project: {CFG.catalog}.{CFG.schema}")
print(f"Model: {CFG.fq_model_name}")
print(f"Endpoint: {CFG.endpoint_name}")

# COMMAND ----------

# DBTITLE 1,Step 1: Ingest
# Run the ingestion notebook
# %run ../src/data/ingestion

# COMMAND ----------

# DBTITLE 1,Step 2: Features
# Run feature engineering
# %run ../src/data/features

# COMMAND ----------

# DBTITLE 1,Step 3: Train
# Run model training
# %run ../src/models/train

# COMMAND ----------

# DBTITLE 1,Step 4: Evaluate & Promote
# Run evaluation
# %run ../src/models/evaluate

# COMMAND ----------

# DBTITLE 1,Step 5: Deploy
# Deploy to serving endpoint
# %run ../src/serving/deploy

# COMMAND ----------

# DBTITLE 1,Step 6: Test Endpoint
# Test the deployed endpoint
import requests, json
from databricks.sdk import WorkspaceClient
from src.config import CFG

w = WorkspaceClient()

# Sample payload
sample = {
    "dataframe_records": [{
        "txn_count_1d": 5, "txn_count_7d": 20,
        "txn_amount_avg_1d": 150.0, "txn_amount_avg_7d": 120.0,
        "txn_amount_std_7d": 45.0, "amount_zscore": 2.1,
        "hour_of_day": 3, "day_of_week": 1, "amount": 500.0
    }]
}

# response = w.serving_endpoints.query(name=CFG.endpoint_name, dataframe_records=sample["dataframe_records"])
# print(response)
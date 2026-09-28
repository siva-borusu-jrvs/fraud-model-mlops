# Databricks notebook source
# DBTITLE 1,Pipeline Config
# Orchestration Pipeline
# This notebook is the entry point for the Databricks Job.
# It chains: Ingest -> Features -> Train -> Evaluate -> Deploy

from src.config import CFG
print(f"Pipeline target: {CFG.catalog}.{CFG.schema}")

# COMMAND ----------

# DBTITLE 1,Step 1: Ingest Data
# MAGIC %run ../src/data/ingestion

# COMMAND ----------

# DBTITLE 1,Run Ingestion
source_path = dbutils.widgets.get("source_path") if "source_path" in [w.name for w in dbutils.widgets.getAll()] else "/Volumes/ml/fraud/landing/transactions/"
checkpoint_path = "/Volumes/ml/fraud/checkpoints/ingestion"

ingest_incremental(source_path, checkpoint_path)
print(f"Ingestion complete: {CFG.fq_raw_table}")

# COMMAND ----------

# DBTITLE 1,Step 2: Feature Engineering
# MAGIC %run ../src/data/features

# COMMAND ----------

# DBTITLE 1,Run Features
features_df = compute_features()
write_feature_table(features_df)
print(f"Features written: {CFG.fq_feature_table}")

# COMMAND ----------

# DBTITLE 1,Step 3: Train Model
# MAGIC %run ../src/models/train

# COMMAND ----------

# DBTITLE 1,Step 4: Evaluate & Promote
# MAGIC %run ../src/models/evaluate

# COMMAND ----------

# DBTITLE 1,Run Evaluation
# The training notebook sets `run` in scope via %run
# compare_and_promote(run.info.run_id)
print("Evaluation complete.")

# COMMAND ----------

# DBTITLE 1,Step 5: Deploy
# MAGIC %run ../src/serving/deploy

# COMMAND ----------

# DBTITLE 1,Run Deployment
deploy_champion()
print("Pipeline complete.")
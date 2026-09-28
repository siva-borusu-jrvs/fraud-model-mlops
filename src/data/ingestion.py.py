# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC Reads raw transaction data into Unity Catalog. Supports batch and incremental (Auto Loader).

# COMMAND ----------

from src.config import CFG

print(f"Target table: {CFG.fq_raw_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Load

# COMMAND ----------

def load_raw_transactions(source_path: str, fmt: str = "parquet"):
    """Read raw transaction files and return a Spark DataFrame."""
    return spark.read.format(fmt).load(source_path)


def write_raw_table(df):
    """Persist raw transactions to the Unity Catalog Delta table."""
    df.write.format("delta").mode("append").saveAsTable(CFG.fq_raw_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incremental Ingestion (Auto Loader)

# COMMAND ----------

def ingest_incremental(source_path: str, checkpoint_path: str):
    """Stream new files into the raw table via Auto Loader."""
    (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{checkpoint_path}/_schema")
        .load(source_path)
        .writeStream
        .format("delta")
        .option("checkpointLocation", checkpoint_path)
        .trigger(availableNow=True)
        .toTable(CFG.fq_raw_table)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run
# MAGIC Uncomment and set paths to execute.

# COMMAND ----------

# source_path = "/Volumes/ml/fraud/landing/transactions/"
# checkpoint_path = "/Volumes/ml/fraud/checkpoints/ingestion"
# ingest_incremental(source_path, checkpoint_path)

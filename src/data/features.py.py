# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering
# MAGIC Builds the `fraud_features` Delta table as the offline feature store.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from src.config import CFG

print(f"Source: {CFG.fq_raw_table}")
print(f"Target: {CFG.fq_feature_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rolling Window Features

# COMMAND ----------

def compute_features():
    """Derive fraud-detection features from raw transactions."""
    raw = spark.table(CFG.fq_raw_table)

    window_1d = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("transaction_ts").cast("long"))
        .rangeBetween(-86400, 0)
    )
    window_7d = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("transaction_ts").cast("long"))
        .rangeBetween(-86400 * 7, 0)
    )

    return (
        raw
        .withColumn("txn_count_1d", F.count("*").over(window_1d))
        .withColumn("txn_count_7d", F.count("*").over(window_7d))
        .withColumn("txn_amount_avg_1d", F.avg("amount").over(window_1d))
        .withColumn("txn_amount_avg_7d", F.avg("amount").over(window_7d))
        .withColumn("txn_amount_std_7d", F.stddev("amount").over(window_7d))
        .withColumn(
            "amount_zscore",
            F.when(F.col("txn_amount_std_7d") > 0,
                (F.col("amount") - F.col("txn_amount_avg_7d")) / F.col("txn_amount_std_7d")
            ).otherwise(0.0),
        )
        .withColumn("hour_of_day", F.hour("transaction_ts"))
        .withColumn("day_of_week", F.dayofweek("transaction_ts"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Feature Table

# COMMAND ----------

def write_feature_table(df):
    """Overwrite the feature Delta table in Unity Catalog."""
    df.write.format("delta").mode("overwrite").saveAsTable(CFG.fq_feature_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# features_df = compute_features()
# features_df.display()

# COMMAND ----------

# write_feature_table(features_df)
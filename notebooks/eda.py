# Databricks notebook source
# DBTITLE 1,Setup
# Exploratory Data Analysis - Fraud Detection
from src.config import CFG
import pyspark.sql.functions as F

print(f"Raw table: {CFG.fq_raw_table}")
print(f"Feature table: {CFG.fq_feature_table}")

# COMMAND ----------

# DBTITLE 1,Data Overview
# MAGIC %sql
# MAGIC -- Quick look at raw data shape and schema
# MAGIC DESCRIBE TABLE ml.fraud.raw_transactions

# COMMAND ----------

# DBTITLE 1,Class Balance
# MAGIC %sql
# MAGIC -- Check fraud vs non-fraud distribution
# MAGIC SELECT
# MAGIC   is_fraud,
# MAGIC   COUNT(*) AS cnt,
# MAGIC   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
# MAGIC FROM ml.fraud.raw_transactions
# MAGIC GROUP BY is_fraud

# COMMAND ----------

# DBTITLE 1,Transaction Amount Distribution
# Amount distribution by fraud label
df = spark.table(CFG.fq_raw_table).select("amount", "is_fraud")
df.display()

# COMMAND ----------

# DBTITLE 1,Temporal Patterns
# MAGIC %sql
# MAGIC -- Hourly fraud rate
# MAGIC SELECT
# MAGIC   HOUR(transaction_ts) AS hour_of_day,
# MAGIC   COUNT(*) AS total_txns,
# MAGIC   SUM(CAST(is_fraud AS INT)) AS fraud_count,
# MAGIC   ROUND(AVG(CAST(is_fraud AS DOUBLE)), 4) AS fraud_rate
# MAGIC FROM ml.fraud.raw_transactions
# MAGIC GROUP BY HOUR(transaction_ts)
# MAGIC ORDER BY hour_of_day

# COMMAND ----------

# DBTITLE 1,Feature Correlations
# Feature correlation matrix
pdf = spark.table(CFG.fq_feature_table).select(
    "txn_count_1d", "txn_count_7d", "txn_amount_avg_1d",
    "txn_amount_avg_7d", "amount_zscore", "amount", "is_fraud"
).toPandas()

pdf.corr().style.background_gradient(cmap="RdBu", vmin=-1, vmax=1)
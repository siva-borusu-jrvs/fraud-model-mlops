# Databricks notebook source
# DBTITLE 1,Synthetic Data for Fraud Detection MLOps Demo
# MAGIC %md
# MAGIC # Synthetic Data for Fraud Detection MLOps Demo
# MAGIC
# MAGIC Generates three tables in `mlops.siva_borusu`:
# MAGIC - **customer_features** — 2,000 card-level aggregate features
# MAGIC - **transactions** — 10,000 transaction events linked to those cards
# MAGIC - **customer_features_drift** — drifted version of customer_features for monitoring demo

# COMMAND ----------

# DBTITLE 1,Imports
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils import get_logger

logger = get_logger("notebooks.synthetic_data")

np.random.seed(42)

# COMMAND ----------

# DBTITLE 1,Generate customer_features table
NUM_CARDS = 2000
US_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
             "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
             "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT"]

card_ids = [f"CC_{i:04d}" for i in range(1, NUM_CARDS + 1)]

customer_features = pd.DataFrame({
    "card_id": card_ids,
    "avg_transaction_amount_90d": np.clip(np.random.normal(150, 80, NUM_CARDS), 10, None).round(2),
    "max_monthly_spend_statement": np.random.uniform(500, 5000, NUM_CARDS).round(2),
    "total_transactions_90d": np.random.randint(10, 201, NUM_CARDS),
    "avg_transactions_per_day_90d": np.random.uniform(0.1, 2.5, NUM_CARDS).round(2),
    "max_single_transaction_90d": np.random.uniform(200, 3000, NUM_CARDS).round(2),
    "pct_online_transactions_90d": np.random.uniform(0.1, 0.9, NUM_CARDS).round(3),
    "home_country": np.random.choice(["US", "CA", "UK", "MX"], NUM_CARDS, p=[0.80, 0.10, 0.05, 0.05]),
    "home_state": np.random.choice(US_STATES, NUM_CARDS),
    "distinct_countries_90d": np.random.choice(range(1, 6), NUM_CARDS, p=[0.50, 0.25, 0.15, 0.07, 0.03]),
    "distinct_states_90d": np.random.choice(range(1, 9), NUM_CARDS, p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]),
    "days_since_last_password_reset": np.random.randint(1, 366, NUM_CARDS),
    "days_since_last_web_login": np.random.randint(0, 31, NUM_CARDS),
    "hours_since_last_transaction": np.random.uniform(0.5, 720, NUM_CARDS).round(1),
    "declined_transactions_30d": np.random.choice(range(0, 6), NUM_CARDS, p=[0.55, 0.25, 0.10, 0.05, 0.03, 0.02]),
})

spark.createDataFrame(customer_features).write.format("delta").mode("overwrite").saveAsTable("mlops.siva_borusu.customer_features")
logger.info(f"customer_features: {NUM_CARDS} rows written")
display(spark.table("mlops.siva_borusu.customer_features").limit(5))

# COMMAND ----------

# DBTITLE 1,Generate transactions table
NUM_TXNS = 10000
FRAUD_RATE = 0.025
MERCHANT_CATS = ["grocery", "travel", "electronics", "restaurant", "gas", "online_retail", "entertainment", "healthcare"]
COUNTRIES = ["US", "CA", "UK", "MX", "DE", "FR"]

# Assign card_ids — some cards get more transactions than others (realistic)
card_weights = np.random.dirichlet(np.ones(NUM_CARDS) * 2, size=1)[0]
txn_card_ids = np.random.choice(card_ids, NUM_TXNS, p=card_weights)

# Base transaction amounts — lognormal for realistic right-skew (most $10-$500, few high)
amounts = np.clip(np.random.lognormal(mean=4.0, sigma=1.0, size=NUM_TXNS), 1, 5000).round(2)

# Timestamps spread across last 90 days
base_time = datetime(2026, 9, 30)
timestamps = [base_time - timedelta(days=np.random.uniform(0, 90), hours=np.random.uniform(0, 24)) for _ in range(NUM_TXNS)]

# Countries and states
txn_countries = np.random.choice(COUNTRIES, NUM_TXNS, p=[0.75, 0.08, 0.05, 0.05, 0.04, 0.03])
txn_states = np.where(
    txn_countries == "US",
    np.random.choice(US_STATES, NUM_TXNS),
    "N/A"
)

# Merchant categories
txn_merchants = np.random.choice(MERCHANT_CATS, NUM_TXNS, p=[0.20, 0.10, 0.12, 0.18, 0.12, 0.15, 0.08, 0.05])

# VPN usage — ~10% baseline
is_vpn = np.random.random(NUM_TXNS) < 0.10

# Transaction status — ~95% approved
txn_status = np.where(np.random.random(NUM_TXNS) < 0.95, "approved", "declined")

# Fraud labels — correlated with high amount, VPN, international, unusual merchant
fraud_score = np.zeros(NUM_TXNS)
fraud_score += (amounts > 1000) * 0.15          # high amounts
fraud_score += is_vpn * 0.20                     # VPN usage
fraud_score += (txn_countries != "US") * 0.10    # international
fraud_score += np.isin(txn_merchants, ["electronics", "online_retail"]) * 0.08  # risky categories
fraud_score += np.random.uniform(0, 0.05, NUM_TXNS)  # noise
is_fraud = fraud_score > np.random.uniform(0.15, 0.45, NUM_TXNS)

# Boost VPN for fraud cases
is_vpn = is_vpn | (is_fraud & (np.random.random(NUM_TXNS) < 0.4))

transactions = pd.DataFrame({
    "transaction_id": [f"TXN_{i:05d}" for i in range(1, NUM_TXNS + 1)],
    "card_id": txn_card_ids,
    "transaction_amount": amounts,
    "transaction_country": txn_countries,
    "transaction_state": txn_states,
    "is_vpn_used": is_vpn,
    "merchant_category": txn_merchants,
    "transaction_status": txn_status,
    "transaction_timestamp": timestamps,
    "is_fraud": is_fraud,
})

spark.createDataFrame(transactions).write.format("delta").mode("overwrite").saveAsTable("mlops.siva_borusu.transactions")
fraud_pct = is_fraud.mean() * 100
logger.info(f"transactions: {NUM_TXNS} rows written | fraud rate: {fraud_pct:.1f}%")
display(spark.table("mlops.siva_borusu.transactions").limit(5))

# COMMAND ----------

# DBTITLE 1,Generate customer_features_drift table
# Start from original customer_features and apply drift
cf_drift = customer_features.copy()

# Shift avg_transaction_amount UP ~40% (mean ~$210)
cf_drift["avg_transaction_amount_90d"] = np.clip(
    np.random.normal(210, 90, NUM_CARDS), 10, None
).round(2)

# Shift max_monthly_spend UP ~30%
cf_drift["max_monthly_spend_statement"] = (cf_drift["max_monthly_spend_statement"] * np.random.uniform(1.2, 1.4, NUM_CARDS)).round(2)

# More online transactions
cf_drift["pct_online_transactions_90d"] = np.clip(
    cf_drift["pct_online_transactions_90d"] + np.random.uniform(0.1, 0.3, NUM_CARDS), 0.1, 0.98
).round(3)

# More international activity
cf_drift["distinct_countries_90d"] = np.clip(
    cf_drift["distinct_countries_90d"] + np.random.choice([0, 1, 2], NUM_CARDS, p=[0.3, 0.4, 0.3]),
    1, 8
).astype(int)

# More recent password resets (account takeover wave)
cf_drift["days_since_last_password_reset"] = np.clip(
    np.random.exponential(30, NUM_CARDS), 1, 365
).astype(int)

# More declines
cf_drift["declined_transactions_30d"] = np.random.choice(
    range(0, 8), NUM_CARDS, p=[0.30, 0.20, 0.18, 0.13, 0.09, 0.05, 0.03, 0.02]
)

# Lower hours_since_last_transaction (more frequent)
cf_drift["hours_since_last_transaction"] = np.clip(
    np.random.uniform(0.5, 400, NUM_CARDS), 0.5, 720
).round(1)

spark.createDataFrame(cf_drift).write.format("delta").mode("overwrite").saveAsTable("mlops.siva_borusu.customer_features_drift")
logger.info(f"customer_features_drift: {NUM_CARDS} rows written")
display(spark.table("mlops.siva_borusu.customer_features_drift").limit(5))

# COMMAND ----------

# DBTITLE 1,Create joined training dataset: customer_n_transactions
txn_df = spark.table("mlops.siva_borusu.transactions")
cf_df = spark.table("mlops.siva_borusu.customer_features")

customer_n_transactions = txn_df.join(cf_df, on="card_id", how="inner")

customer_n_transactions.write.format("delta").mode("overwrite").saveAsTable("mlops.siva_borusu.customer_n_transactions")

logger.info(f"customer_n_transactions: {customer_n_transactions.count()} rows, {len(customer_n_transactions.columns)} columns")
display(spark.table("mlops.siva_borusu.customer_n_transactions").limit(5))

# COMMAND ----------

# DBTITLE 1,Validate tables
# --- Row counts ---
for t in ["customer_features", "transactions", "customer_features_drift"]:
    cnt = spark.table(f"mlops.siva_borusu.{t}").count()
    logger.info(f"{t}: {cnt} rows")

# --- Orphan check: all transaction card_ids exist in customer_features ---
txn_cards = set(spark.table("mlops.siva_borusu.transactions").select("card_id").distinct().toPandas()["card_id"])
cf_cards = set(spark.table("mlops.siva_borusu.customer_features").select("card_id").distinct().toPandas()["card_id"])
orphans = txn_cards - cf_cards
logger.info(f"Orphan card_ids in transactions: {len(orphans)} {'PASS' if len(orphans) == 0 else 'FAIL'}")

# --- Drift comparison: mean values side by side ---
import pandas as pd

drift_cols = ["avg_transaction_amount_90d", "max_monthly_spend_statement", "pct_online_transactions_90d",
              "distinct_countries_90d", "days_since_last_password_reset", "declined_transactions_30d", "hours_since_last_transaction"]

cf_base = spark.table("mlops.siva_borusu.customer_features").select(drift_cols).toPandas()
cf_drft = spark.table("mlops.siva_borusu.customer_features_drift").select(drift_cols).toPandas()

comparison = pd.DataFrame({
    "column": drift_cols,
    "baseline_mean": [cf_base[c].mean() for c in drift_cols],
    "drift_mean": [cf_drft[c].mean() for c in drift_cols],
})
comparison["shift_%"] = ((comparison["drift_mean"] - comparison["baseline_mean"]) / comparison["baseline_mean"] * 100).round(1)
logger.info("Drift Comparison (baseline vs drift):")
display(comparison)

# COMMAND ----------

# DBTITLE 1,Generate customer_n_transactions_drift with model predictions
# ── Generate drifted transactions, join with drifted customer features,
#    score through the champion model, and save as the drift monitoring table.

import mlflow
from pyspark.sql.functions import struct, lit, hour, dayofweek, col

NUM_DRIFT_TXNS = 10_000
np.random.seed(99)

# Card assignment — same card pool as customer_features_drift
drift_card_weights = np.random.dirichlet(np.ones(NUM_CARDS) * 2, size=1)[0]
drift_txn_card_ids = np.random.choice(card_ids, NUM_DRIFT_TXNS, p=drift_card_weights)

# Drifted amounts — higher mean (~$230 vs original ~$150)
drift_amounts = np.clip(
    np.random.lognormal(mean=4.5, sigma=1.1, size=NUM_DRIFT_TXNS), 1, 8000
).round(2)

# Recent timestamps (last 30 days)
drift_base_time = datetime(2026, 10, 4)
drift_timestamps = [
    drift_base_time - timedelta(
        days=np.random.uniform(0, 30), hours=np.random.uniform(0, 24)
    )
    for _ in range(NUM_DRIFT_TXNS)
]

# More international transactions (US drops from 75% to 55%)
drift_txn_countries = np.random.choice(
    COUNTRIES, NUM_DRIFT_TXNS, p=[0.55, 0.15, 0.10, 0.08, 0.07, 0.05]
)
drift_txn_states = np.where(
    drift_txn_countries == "US",
    np.random.choice(US_STATES, NUM_DRIFT_TXNS),
    "N/A",
)

# More electronics / online_retail (risky categories)
drift_txn_merchants = np.random.choice(
    MERCHANT_CATS, NUM_DRIFT_TXNS,
    p=[0.12, 0.10, 0.20, 0.12, 0.08, 0.22, 0.08, 0.08],
)

# Higher VPN usage (10% -> 20%)
drift_is_vpn = np.random.random(NUM_DRIFT_TXNS) < 0.20

# More declines (5% -> 12%)
drift_txn_status = np.where(
    np.random.random(NUM_DRIFT_TXNS) < 0.88, "approved", "declined"
)

# Higher fraud rate — concept drift (shifted thresholds)
drift_fraud_score = np.zeros(NUM_DRIFT_TXNS)
drift_fraud_score += (drift_amounts > 800) * 0.18
drift_fraud_score += drift_is_vpn * 0.25
drift_fraud_score += (drift_txn_countries != "US") * 0.15
drift_fraud_score += np.isin(drift_txn_merchants, ["electronics", "online_retail"]) * 0.12
drift_fraud_score += np.random.uniform(0, 0.08, NUM_DRIFT_TXNS)
drift_is_fraud = drift_fraud_score > np.random.uniform(0.12, 0.40, NUM_DRIFT_TXNS)
drift_is_vpn = drift_is_vpn | (drift_is_fraud & (np.random.random(NUM_DRIFT_TXNS) < 0.5))

drift_transactions = pd.DataFrame({
    "transaction_id": [f"DTXN_{i:05d}" for i in range(1, NUM_DRIFT_TXNS + 1)],
    "card_id": drift_txn_card_ids,
    "transaction_amount": drift_amounts,
    "transaction_country": drift_txn_countries,
    "transaction_state": drift_txn_states,
    "is_vpn_used": drift_is_vpn,
    "merchant_category": drift_txn_merchants,
    "transaction_status": drift_txn_status,
    "transaction_timestamp": drift_timestamps,
    "is_fraud": drift_is_fraud,
})

# ── Join drifted transactions with drifted customer features ──────
drift_txn_df = spark.createDataFrame(drift_transactions)
drift_cf_df = spark.table("mlops.siva_borusu.customer_features_drift")
customer_n_txn_drift = (
    drift_txn_df.join(drift_cf_df, on="card_id", how="inner")
    .withColumn("hour_of_day", hour("transaction_timestamp"))
    .withColumn("day_of_week", dayofweek("transaction_timestamp"))
    .withColumn("is_vpn_used", col("is_vpn_used").cast("long"))
    .withColumn("is_fraud", col("is_fraud").cast("long"))
)

# ── Score through the champion model ──────────────────────────────
MODEL_URI = "models:/mlops.siva_borusu.fraud_model_credit_card@champion"
predict_udf = mlflow.pyfunc.spark_udf(
    spark, model_uri=MODEL_URI, result_type="double"
)

exclude_cols = {"transaction_id", "card_id", "transaction_timestamp", "is_fraud"}
feature_cols = [c for c in customer_n_txn_drift.columns if c not in exclude_cols]

scored_drift = (
    customer_n_txn_drift
    .withColumn("prediction", predict_udf(struct(*feature_cols)))
    .withColumn("model_version", lit(MODEL_URI))
)

scored_drift.write.format("delta").mode("overwrite").saveAsTable(
    "mlops.siva_borusu.customer_n_transactions_drift"
)

drift_fraud_pct = drift_is_fraud.mean() * 100
logger.info(
    f"customer_n_transactions_drift: {NUM_DRIFT_TXNS} rows "
    f"| actual fraud rate: {drift_fraud_pct:.1f}%"
)
display(spark.table("mlops.siva_borusu.customer_n_transactions_drift").limit(5))

# COMMAND ----------

# DBTITLE 1,Scored Baseline for Drift Monitor
# MAGIC %md
# MAGIC ## Scored Baseline for Drift Monitor
# MAGIC The InferenceLog monitor requires `prediction` and `model_version` columns on the baseline
# MAGIC table too, so that it can compute identical metric sets on both sides and diff them.
# MAGIC This cell scores `customer_n_transactions` through the **same champion model** used in
# MAGIC cell 8 above and saves the result as `customer_n_transactions_baseline`.

# COMMAND ----------

# DBTITLE 1,Generate customer_n_transactions_baseline with model predictions
# ── Score the training data through the same champion model to create
#    a baseline table compatible with the InferenceLog drift monitor.

baseline_df = (
    spark.table("mlops.siva_borusu.customer_n_transactions")
    .withColumn("hour_of_day", hour("transaction_timestamp"))
    .withColumn("day_of_week", dayofweek("transaction_timestamp"))
    .withColumn("is_vpn_used", col("is_vpn_used").cast("long"))
    .withColumn("is_fraud", col("is_fraud").cast("double"))
)

# Same model URI as cell 8
baseline_feature_cols = [c for c in baseline_df.columns if c not in exclude_cols]

scored_baseline = (
    baseline_df
    .withColumn("prediction", predict_udf(struct(*baseline_feature_cols)))
    .withColumn("model_version", lit(MODEL_URI))
)

scored_baseline.write.format("delta").mode("overwrite").saveAsTable(
    "mlops.siva_borusu.customer_n_transactions_baseline"
)

logger.info(
    f"customer_n_transactions_baseline: {scored_baseline.count()} rows "
    f"| scored with {MODEL_URI}"
)
display(spark.table("mlops.siva_borusu.customer_n_transactions_baseline").limit(5))
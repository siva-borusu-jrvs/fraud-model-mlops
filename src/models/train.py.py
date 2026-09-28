# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training
# MAGIC Trains a fraud classifier, logs to MLflow, registers to Unity Catalog.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.config import CFG

mlflow.set_experiment(CFG.experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

FEATURE_COLS = [
    "txn_count_1d", "txn_count_7d", "txn_amount_avg_1d", "txn_amount_avg_7d",
    "txn_amount_std_7d", "amount_zscore", "hour_of_day", "day_of_week", "amount",
]
LABEL_COL = "is_fraud"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

pdf = (
    spark.table(CFG.fq_feature_table)
    .select(FEATURE_COLS + [LABEL_COL])
    .toPandas()
    .dropna()
)

X = pdf[FEATURE_COLS]
y = pdf[LABEL_COL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Fraud rate: {y.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train & Log

# COMMAND ----------

with mlflow.start_run() as run:
    params = dict(n_estimators=200, max_depth=6, learning_rate=0.1)
    mlflow.log_params(params)

    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=CFG.fq_model_name, input_example=X_test.head(5))
    print(f"Run ID: {run.info.run_id}")
    print(f"Metrics: {metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

import pandas as pd
pd.DataFrame([metrics], index=["GBT"]).T

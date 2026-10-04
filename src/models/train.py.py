# Databricks notebook source
# DBTITLE 1,Model Training
# MAGIC %md
# MAGIC # Model Training — Credit Card Fraud Detection
# MAGIC Builds a sklearn Pipeline (ColumnTransformer + LightGBM) with the best hyperparameters
# MAGIC from Optuna tuning. Logs metrics to MLflow and registers the model to Unity Catalog.
# MAGIC For hyperparameter tuning, see `train_optuna.py`.

# COMMAND ----------

# DBTITLE 1,Imports and setup
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from mlflow.models import infer_signature
from lightgbm import LGBMClassifier
from src.config import CFG
from src.utils import get_logger

logger = get_logger("models.train")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CFG.experiment_name)

# COMMAND ----------

# DBTITLE 1,Configuration
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# DBTITLE 1,Feature definitions and best params
LABEL_COL = "is_fraud"
DROP_COLS = ["card_id", "transaction_id", "transaction_timestamp"]

CATEGORICAL_COLS = [
    "transaction_country", "transaction_state", "merchant_category",
    "transaction_status", "home_country", "home_state",
]

NUMERIC_COLS = [
    "transaction_amount", "avg_transaction_amount_90d", "max_monthly_spend_statement",
    "total_transactions_90d", "avg_transactions_per_day_90d", "max_single_transaction_90d",
    "pct_online_transactions_90d", "distinct_countries_90d", "distinct_states_90d",
    "days_since_last_password_reset", "days_since_last_web_login",
    "hours_since_last_transaction", "declined_transactions_30d",
    "is_vpn_used", "hour_of_day", "day_of_week",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# Best parameters from Optuna tuning (train_optuna.py)
BEST_PARAMS = {
    "n_estimators": 347,
    "max_depth": 3,
    "learning_rate": 0.049,
    "num_leaves": 36,
    "min_child_samples": 24,
    "subsample": 0.800,
    "colsample_bytree": 0.826,
}

# COMMAND ----------

# DBTITLE 1,Load data
# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# DBTITLE 1,Load and preprocess data
pdf = spark.table(CFG.fq_feature_table).toPandas()

pdf["hour_of_day"] = pd.to_datetime(pdf["transaction_timestamp"]).dt.hour
pdf["day_of_week"] = pd.to_datetime(pdf["transaction_timestamp"]).dt.dayofweek
pdf["is_vpn_used"] = pdf["is_vpn_used"].astype(int)
pdf = pdf.drop(columns=DROP_COLS)

X = pdf[FEATURE_COLS]
y = pdf[LABEL_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Fraud rate: {y.mean():.4f}")
logger.info(f"Features: {len(FEATURE_COLS)} ({len(CATEGORICAL_COLS)} categorical, {len(NUMERIC_COLS)} numeric)")

# COMMAND ----------

# DBTITLE 1,Train model
# MAGIC %md
# MAGIC ## Train and Register Model

# COMMAND ----------

# DBTITLE 1,Train, evaluate, log, and register
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(**BEST_PARAMS, random_state=42, verbose=-1)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "test_f1": f1_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_recall": recall_score(y_test, y_pred),
    "test_roc_auc": roc_auc_score(y_test, y_prob),
}

with mlflow.start_run(run_name="lgbm_best_params") as run:
    mlflow.log_params(BEST_PARAMS)
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name=CFG.fq_model_name,
        input_example=X_test.head(5),
        signature=signature,
    )

    logger.info(f"Test metrics: {metrics}")
    logger.info(f"Run ID: {run.info.run_id}")
    logger.info(f"Model registered: {CFG.fq_model_name}")

# COMMAND ----------

# DBTITLE 1,Results
# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# DBTITLE 1,Classification report
logger.info(f"\n{classification_report(y_test, y_pred, target_names=['legit', 'fraud'])}")
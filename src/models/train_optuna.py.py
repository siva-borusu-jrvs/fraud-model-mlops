# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Model Training
# MAGIC %md
# MAGIC # Model Training with Optuna — Credit Card Fraud Detection
# MAGIC Builds a sklearn Pipeline (ColumnTransformer + LightGBM) with Optuna hyperparameter tuning.
# MAGIC Logs all trials to MLflow, registers the best model to Unity Catalog.
# MAGIC Use `train.py` for a simplified version without tuning.

# COMMAND ----------

# DBTITLE 1,Imports and experiment setup
import mlflow
import mlflow.sklearn
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# DBTITLE 1,Feature definitions
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# DBTITLE 1,Load and preprocess data
pdf = spark.table(CFG.fq_feature_table).toPandas()

# Extract time features from timestamp
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

# DBTITLE 1,Pipeline & Hyperparameter Tuning
# MAGIC %md
# MAGIC ## Pipeline & Hyperparameter Tuning

# COMMAND ----------

# DBTITLE 1,Optuna tuning with MLflow logging
# --- Build the preprocessing + model pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS),
    ]
)


def build_pipeline(params):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(**params, random_state=42, verbose=-1)),
    ])


# --- Optuna objective: 3-fold CV on ROC-AUC ---
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    pipe = build_pipeline(params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall")

    # Log each trial as a nested MLflow run
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metric("cv_recall_mean", scores.mean())
        mlflow.log_metric("cv_recall_std", scores.std())

    return scores.mean()


# --- Run tuning under a parent MLflow run ---
with mlflow.start_run(run_name="lgbm_optuna_tuning") as parent_run:
    study = optuna.create_study(direction="maximize", study_name="fraud_lgbm")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # --- Refit best pipeline on full training set ---
    best_params = study.best_params
    best_pipeline = build_pipeline(best_params)
    best_pipeline.fit(X_train, y_train)

    # --- Evaluate on held-out test set ---
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_f1": f1_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob),
        "best_cv_recall": study.best_value,
    }

    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metrics(metrics)

    # --- Log the full pipeline and register to UC ---
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        best_pipeline,
        artifact_path="model",
        registered_model_name=CFG.fq_model_name,
        input_example=X_test.head(5),
        signature=signature,
    )

    logger.info(f"Best trial: #{study.best_trial.number} | CV Recall: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")
    logger.info(f"Test metrics: {metrics}")
    logger.info(f"Parent Run ID: {parent_run.info.run_id}")
    logger.info(f"Model registered to: {CFG.fq_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# DBTITLE 1,Results summary
logger.info(f"\n{classification_report(y_test, y_pred, target_names=['legit', 'fraud'])}")

# Show top 10 trials
trials_df = study.trials_dataframe()[["number", "value", "params_n_estimators", "params_max_depth", "params_learning_rate", "params_num_leaves"]]
trials_df = trials_df.sort_values("value", ascending=False).head(10).rename(columns={"value": "cv_recall"})
display(trials_df)
# Databricks notebook source
# DBTITLE 1,Feature Lookup Training
# MAGIC %md
# MAGIC # Feature Lookup Training — Credit Card Fraud Detection
# MAGIC Retrains the fraud model with `FeatureLookup` (customer features via online table) and
# MAGIC `FeatureFunction` (time features computed at serving time). The logged model receives a
# MAGIC lightweight transaction payload and resolves everything else automatically.

# COMMAND ----------

# DBTITLE 1,Imports and setup
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier
from pyspark.sql.functions import col
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction
from src.config import CFG
from src.utils import get_logger

logger = get_logger("models.train_featurelookup")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CFG.experiment_name)
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Set up feature table
# MAGIC %md
# MAGIC ## Set Up Feature Table
# MAGIC Add a primary key constraint on `customer_features` so it qualifies as a feature table
# MAGIC for online serving.

# COMMAND ----------

# DBTITLE 1,Add primary key constraint
# Make customer_features a feature table by adding a primary key
try:
    spark.sql(f"ALTER TABLE {CFG.fq_customer_features_table} ALTER COLUMN card_id SET NOT NULL")
    spark.sql(
        f"ALTER TABLE {CFG.fq_customer_features_table} "
        f"ADD CONSTRAINT customer_features_pk PRIMARY KEY (card_id)"
    )
    logger.info(f"Primary key added: {CFG.fq_customer_features_table}.card_id")
except Exception as e:
    if "already exists" in str(e).lower() or "CONSTRAINT_ALREADY_EXISTS" in str(e):
        logger.info(f"Primary key already exists on {CFG.fq_customer_features_table}.card_id")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Feature Functions
# MAGIC %md
# MAGIC ## Feature Functions
# MAGIC Register UC Python functions that compute `hour_of_day` and `day_of_week` from
# MAGIC `transaction_timestamp` at serving time — no pyfunc wrapper needed.

# COMMAND ----------

# DBTITLE 1,Create UC functions for time features
spark.sql(f"""
CREATE OR REPLACE FUNCTION {CFG.catalog}.{CFG.schema}.extract_hour_of_day(ts TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON AS $$
return ts.hour
$$
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CFG.catalog}.{CFG.schema}.extract_day_of_week(ts TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON AS $$
return ts.weekday()
$$
""")

logger.info(f"UC functions created: {CFG.catalog}.{CFG.schema}.extract_hour_of_day, extract_day_of_week")

# COMMAND ----------

# DBTITLE 1,Training set
# MAGIC %md
# MAGIC ## Build Training Set
# MAGIC The labels DataFrame contains only raw transaction fields. `FeatureLookup` resolves the
# MAGIC 14 customer features from the feature table via `card_id`. `FeatureFunction` computes the
# MAGIC 2 time features from `transaction_timestamp`. The model sees all 22 features.

# COMMAND ----------

# DBTITLE 1,Define lookups, functions, and create training set
# Labels DataFrame: raw transaction fields + lookup key + label
labels_df = (
    spark.table(CFG.fq_raw_table)
    .withColumn("is_vpn_used", col("is_vpn_used").cast("int"))
    .select(
        "card_id", "transaction_timestamp",
        "transaction_amount", "transaction_country", "transaction_state",
        "merchant_category", "transaction_status", "is_vpn_used",
        "is_fraud",
    )
)

# FeatureLookup: resolve 14 customer features from card_id
feature_lookups = [
    FeatureLookup(
        table_name=CFG.fq_customer_features_table,
        lookup_key="card_id",
    ),
]

# FeatureFunction: compute time features from transaction_timestamp
feature_functions = [
    FeatureFunction(
        udf_name=f"{CFG.catalog}.{CFG.schema}.extract_hour_of_day",
        input_bindings={"ts": "transaction_timestamp"},
        output_name="hour_of_day",
    ),
    FeatureFunction(
        udf_name=f"{CFG.catalog}.{CFG.schema}.extract_day_of_week",
        input_bindings={"ts": "transaction_timestamp"},
        output_name="day_of_week",
    ),
]

# Create training set — card_id and transaction_timestamp are consumed by
# lookups/functions but excluded from the final feature set
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups + feature_functions,
    label="is_fraud",
    exclude_columns=["card_id", "transaction_timestamp"],
)

training_df = training_set.load_df().toPandas()
logger.info(f"Training set: {len(training_df):,} rows, {len(training_df.columns)} columns")
logger.info(f"Columns: {sorted(training_df.columns.tolist())}")

# COMMAND ----------

# DBTITLE 1,Train model
# MAGIC %md
# MAGIC ## Train Model
# MAGIC Same sklearn Pipeline as `train.py` using the best hyperparameters from Optuna tuning.
# MAGIC No re-tuning — deterministic rebuild.

# COMMAND ----------

# DBTITLE 1,Train with best params from train.py
LABEL_COL = "is_fraud"

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

# Best parameters from Optuna tuning in train.py
BEST_PARAMS = {
    "n_estimators": 347,
    "max_depth": 3,
    "learning_rate": 0.049,
    "num_leaves": 36,
    "min_child_samples": 24,
    "subsample": 0.800,
    "colsample_bytree": 0.826,
}

X = training_df[FEATURE_COLS]
y = training_df[LABEL_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Fraud rate: {y.mean():.4f}")

# Same pipeline as train.py
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

# Evaluate on held-out test set
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "test_f1": f1_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_recall": recall_score(y_test, y_pred),
    "test_roc_auc": roc_auc_score(y_test, y_prob),
}
logger.info(f"Test metrics: {metrics}")

# COMMAND ----------

# DBTITLE 1,Log model
# MAGIC %md
# MAGIC ## Log Model with Feature Lineage
# MAGIC `fe.log_model` records which features the model needs so the serving endpoint
# MAGIC automatically resolves them via FeatureLookup (online table) and FeatureFunction (UC UDFs).

# COMMAND ----------

# DBTITLE 1,Log model with fe.log_model
with mlflow.start_run(run_name="lgbm_feature_lookup") as run:
    mlflow.log_params(BEST_PARAMS)
    mlflow.log_metrics(metrics)

    fe.log_model(
        model=pipeline,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=f"{CFG.catalog}.{CFG.schema}.fraud_model_credit_card_featurestore",
        input_example=X_test.head(5),
    )

    logger.info(f"Run ID: {run.info.run_id}")
    logger.info(f"Model registered: {CFG.catalog}.{CFG.schema}.fraud_model_credit_card_featurestore (with FeatureLookup + FeatureFunction lineage)")

# COMMAND ----------

# DBTITLE 1,Results
# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# DBTITLE 1,Classification report
logger.info(f"\n{classification_report(y_test, y_pred, target_names=['legit', 'fraud'])}")
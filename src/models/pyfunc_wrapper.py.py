# Databricks notebook source
# DBTITLE 1,PyFunc Wrapper Model
# MAGIC %md
# MAGIC # PyFunc Wrapper Model — Credit Card Fraud Detection
# MAGIC Wraps the trained sklearn pipeline in a custom `mlflow.pyfunc.PythonModel` that receives
# MAGIC a lightweight transaction payload, looks up customer features from the Delta table,
# MAGIC computes time features, and returns a fraud prediction. No Feature Store or online table required.

# COMMAND ----------

# DBTITLE 1,Imports and setup
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
from src.config import CFG
from src.utils import get_logger

logger = get_logger("models.pyfunc_wrapper")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CFG.experiment_name)

# COMMAND ----------

# DBTITLE 1,Define wrapper class
# MAGIC %md
# MAGIC ## PyFunc Wrapper Class
# MAGIC The wrapper receives 8 raw transaction fields, looks up customer features from the
# MAGIC Delta table (loaded once at startup via Databricks SDK + Statement Execution API),
# MAGIC computes time features, and calls the inner sklearn pipeline.

# COMMAND ----------

# DBTITLE 1,FraudModelWrapper class
class FraudModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom pyfunc that wraps the fraud sklearn pipeline.

    At startup (load_context):
      - Loads the inner sklearn model from artifacts
      - Connects to the Databricks workspace via SQL connector
      - Loads customer_features from the live Delta table

    At predict time:
      - Receives: card_id, transaction_timestamp, transaction_amount,
        transaction_country, transaction_state, merchant_category,
        transaction_status, is_vpn_used
      - Looks up customer features by card_id (cached in memory)
      - Computes hour_of_day and day_of_week from transaction_timestamp
      - Assembles all 22 features and calls the inner model
    """

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

    CUSTOMER_FEATURES_TABLE = "mlops.siva_borusu.customer_features"
    WAREHOUSE_ID = "960eea0736a2e75b"

    def load_context(self, context):
        """Load the inner model and customer features from Delta table."""
        self.model = mlflow.sklearn.load_model(context.artifacts["sklearn_model"])
        self._load_customer_features()

    def _load_customer_features(self):
        """Load customer features using Databricks SDK with OAuth M2M auto-auth.

        No manual env vars needed — the serving endpoint provisions credentials
        automatically via the model's declared resources (DatabricksSQLWarehouse
        + DatabricksTable passed to mlflow.pyfunc.log_model).
        """
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        stmt = w.statement_execution.execute_statement(
            warehouse_id=self.WAREHOUSE_ID,
            statement=f"SELECT * FROM {self.CUSTOMER_FEATURES_TABLE}",
            wait_timeout="30s",
        )

        # Build DataFrame and cast types from the response
        cols = [c.name for c in stmt.manifest.schema.columns]
        col_types = {c.name: c.type_name for c in stmt.manifest.schema.columns}
        self.customer_features = pd.DataFrame(stmt.result.data_array, columns=cols)

        for name, dtype in col_types.items():
            if dtype in ("DOUBLE", "FLOAT", "DECIMAL"):
                self.customer_features[name] = pd.to_numeric(
                    self.customer_features[name], errors="coerce"
                )
            elif dtype in ("INT", "LONG", "SHORT", "BIGINT", "TINYINT", "SMALLINT"):
                self.customer_features[name] = pd.to_numeric(
                    self.customer_features[name], errors="coerce"
                ).astype("Int64")

    def predict(self, context, model_input, params=None):
        """Look up features, compute time features, and predict."""
        df = model_input.copy()

        # Compute time features from transaction_timestamp
        ts = pd.to_datetime(df["transaction_timestamp"])
        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_vpn_used"] = df["is_vpn_used"].astype(int)

        # Look up customer features by card_id
        df = df.merge(self.customer_features, on="card_id", how="left")

        # Select features in the expected order and predict
        return self.model.predict(df[self.FEATURE_COLS])

# COMMAND ----------

# DBTITLE 1,Log wrapper model
# MAGIC %md
# MAGIC ## Log and Register Wrapper Model
# MAGIC Load the champion model from UC, snapshot the customer features table,
# MAGIC and log the pyfunc wrapper with both as artifacts.

# COMMAND ----------

# DBTITLE 1,Save artifacts and log pyfunc
import tempfile, os
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

# Define the serving input/output signature
input_schema = Schema([
    ColSpec("string", "card_id"),
    ColSpec("string", "transaction_timestamp"),
    ColSpec("double", "transaction_amount"),
    ColSpec("string", "transaction_country"),
    ColSpec("string", "transaction_state"),
    ColSpec("string", "merchant_category"),
    ColSpec("string", "transaction_status"),
    ColSpec("long", "is_vpn_used"),
])
output_schema = Schema([ColSpec("long", "prediction")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Input example — one raw transaction payload, exactly what a caller sends.
# MLflow stores this alongside the model so the Serving UI "Query endpoint"
# tab is pre-filled with a valid request body.
input_example = pd.DataFrame([{
    "card_id": "CC_0001",
    "transaction_timestamp": "2026-09-30 03:15:00",
    "transaction_amount": 2500.00,
    "transaction_country": "US",
    "transaction_state": "CA",
    "merchant_category": "electronics",
    "transaction_status": "approved",
    "is_vpn_used": 1,
}])

# Load the champion model from UC and save as local artifact
champion_uri = CFG.champion_model_uri
logger.info(f"Loading inner model from: {champion_uri}")
inner_model = mlflow.sklearn.load_model(champion_uri)

tmp_dir = tempfile.mkdtemp()
inner_model_path = os.path.join(tmp_dir, "sklearn_model")
mlflow.sklearn.save_model(inner_model, inner_model_path)

# Declare resources so serving auto-provisions OAuth M2M access
from mlflow.models.resources import DatabricksSQLWarehouse, DatabricksTable

resources = [
    DatabricksSQLWarehouse(warehouse_id=CFG.sql_warehouse_id),
    DatabricksTable(table_name=CFG.fq_customer_features_table),
]

# Log the pyfunc wrapper model — no parquet artifact,
# customer features loaded from live Delta table at runtime
with mlflow.start_run(run_name="fraud_pyfunc_wrapper") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FraudModelWrapper(),
        artifacts={"sklearn_model": inner_model_path},
        registered_model_name=f"{CFG.catalog}.{CFG.schema}.fraud_model_credit_card_pyfunc",
        signature=signature,
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            "mlflow", "lightgbm", "scikit-learn", "pandas", "pyarrow",
            "databricks-sdk",
        ],
    )
    # Copy training metrics from the inner sklearn model so downstream
    # notebooks (evaluate.py, ab_test.py) can look up metrics by pyfunc version.
    _sklearn_model = f"{CFG.catalog}.{CFG.schema}.{CFG.registered_model_name}"
    try:
        from mlflow.tracking import MlflowClient as _MC
        _mc = _MC()
        _inner_mv = _mc.get_model_version_by_alias(_sklearn_model, "champion")
        _inner_metrics = _mc.get_run(_inner_mv.run_id).data.metrics
        for k, v in _inner_metrics.items():
            mlflow.log_metric(k, v)
        logger.info(f"Copied {len(_inner_metrics)} training metrics from sklearn model v{_inner_mv.version}")
    except Exception as e:
        logger.warning(f"Could not copy training metrics: {e}")

    logger.info(f"Run ID: {run.info.run_id}")
    logger.info(f"Wrapper model registered: {CFG.catalog}.{CFG.schema}.fraud_model_credit_card_pyfunc")

# Always alias the newly registered version as @challenger
from mlflow.tracking import MlflowClient
_client = MlflowClient()
_pyfunc_model_name = f"{CFG.catalog}.{CFG.schema}.fraud_model_credit_card_pyfunc"
_new_version = model_info.registered_model_version
_client.set_registered_model_alias(_pyfunc_model_name, "challenger", _new_version)
logger.info(f"Version {_new_version} aliased as @challenger")

# COMMAND ----------

# DBTITLE 1,Test wrapper
# MAGIC %md
# MAGIC ## Test Locally
# MAGIC Test the predict logic directly using Spark for the feature lookup (notebook context).
# MAGIC At serving time, `load_context` uses the Databricks SDK with OAuth M2M auto-auth.

# COMMAND ----------

# DBTITLE 1,Test with sample input
# Test the wrapper directly (SQL connector not needed in notebook context)
wrapper = FraudModelWrapper()
wrapper.model = inner_model
wrapper.customer_features = spark.table(CFG.fq_customer_features_table).toPandas()

# Sample transaction payload — what the caller would send
test_payload = pd.DataFrame([{
    "card_id": "CC_0001",
    "transaction_timestamp": "2026-09-30 03:15:00",
    "transaction_amount": 2500.00,
    "transaction_country": "US",
    "transaction_state": "CA",
    "merchant_category": "electronics",
    "transaction_status": "approved",
    "is_vpn_used": 1,
}])

prediction = wrapper.predict(None, test_payload)
logger.info(f"Payload: card_id=CC_0001, amount=2500, VPN=True, 3AM electronics purchase")
logger.info(f"Prediction: {'FRAUD' if prediction[0] == 1 else 'LEGIT'} ({prediction[0]})")

# COMMAND ----------

# DBTITLE 1,Endpoint request format
# MAGIC %md
# MAGIC ## Endpoint Request Format
# MAGIC The signature is a column-based schema, so the model expects a **pandas DataFrame**. Model Serving
# MAGIC accepts it as `dataframe_records` (list of row dicts) or `dataframe_split` (columns + data). A bare
# MAGIC dict like the one above is **not** a valid body — it has to be wrapped.
# MAGIC
# MAGIC **Preferred — `dataframe_records`:** two rows for contrast — a 3 AM $2,500 electronics purchase
# MAGIC over VPN (should lean fraud) and a mid-day $43 grocery run (should lean legit).
# MAGIC ```json
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {
# MAGIC       "card_id": "CC_0001",
# MAGIC       "transaction_timestamp": "2026-09-30 03:15:00",
# MAGIC       "transaction_amount": 2500.0,
# MAGIC       "transaction_country": "US",
# MAGIC       "transaction_state": "CA",
# MAGIC       "merchant_category": "electronics",
# MAGIC       "transaction_status": "approved",
# MAGIC       "is_vpn_used": 1
# MAGIC     },
# MAGIC     {
# MAGIC       "card_id": "CC_0002",
# MAGIC       "transaction_timestamp": "2026-09-30 13:40:00",
# MAGIC       "transaction_amount": 42.75,
# MAGIC       "transaction_country": "US",
# MAGIC       "transaction_state": "CA",
# MAGIC       "merchant_category": "grocery",
# MAGIC       "transaction_status": "approved",
# MAGIC       "is_vpn_used": 0
# MAGIC     }
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Alternative — `dataframe_split`** (more compact for batches):
# MAGIC ```json
# MAGIC {
# MAGIC   "dataframe_split": {
# MAGIC     "columns": ["card_id", "transaction_timestamp", "transaction_amount", "transaction_country",
# MAGIC                 "transaction_state", "merchant_category", "transaction_status", "is_vpn_used"],
# MAGIC     "data": [["CC_0001", "2026-09-30 03:15:00", 2500.0, "US", "CA", "electronics", "approved", 1]]
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Response:** one element per input row, in the same order.
# MAGIC ```json
# MAGIC {"predictions": [1, 0]}
# MAGIC ```
# MAGIC
# MAGIC Notes:
# MAGIC * Add or remove dicts in `dataframe_records` to score a batch in one call.
# MAGIC * `is_vpn_used` must be an integer (`0`/`1`), not `true`/`false` — the signature declares it `long`.
# MAGIC * `transaction_timestamp` is a string; `pd.to_datetime` parses it inside `predict`.
# MAGIC * Send **only** these 8 fields — customer features are looked up server-side by `card_id`.
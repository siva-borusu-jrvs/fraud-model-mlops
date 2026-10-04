# Databricks notebook source
# DBTITLE 1,Model Evaluation with MLflow Evaluate
# MAGIC %md
# MAGIC # Model Evaluation with MLflow Evaluate
# MAGIC Enhanced offline validation of the challenger model using `mlflow.evaluate()`.
# MAGIC Automates classification metrics, confusion matrix, ROC curve, PR curve, and per-class
# MAGIC reporting — all logged as MLflow artifacts for cross-run comparison in the UI.
# MAGIC
# MAGIC Replaces the manual threshold check in `evaluate.py` while keeping the same quality gate.

# COMMAND ----------

# DBTITLE 1,Imports and config
import mlflow
import mlflow.models
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from src.config import CFG
from src.utils import get_logger

logger = get_logger("models.evaluate_mlflow")
client = MlflowClient(registry_uri="databricks-uc")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CFG.experiment_name)

_pyfunc_model = f"{CFG.fq_model_name}_pyfunc"
_sklearn_model = CFG.fq_model_name
logger.info(f"Pyfunc model: {_pyfunc_model}")
logger.info(f"Sklearn model: {_sklearn_model}")
logger.info(f"Mode: MLflow Evaluate (full classification report)")

# COMMAND ----------

# DBTITLE 1,Feature definitions
# ── Must match train.py / train_optuna.py exactly ────────────────────
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

# DBTITLE 1,Load and preprocess data
# ── Same data prep as train.py — identical split for reproducibility ──
pdf = spark.table(CFG.fq_feature_table).toPandas()

# Derive time features (same as training)
pdf["hour_of_day"] = pd.to_datetime(pdf["transaction_timestamp"]).dt.hour
pdf["day_of_week"] = pd.to_datetime(pdf["transaction_timestamp"]).dt.dayofweek
pdf["is_vpn_used"] = pdf["is_vpn_used"].astype(int)
pdf = pdf.drop(columns=DROP_COLS)

X = pdf[FEATURE_COLS]
y = pdf[LABEL_COL].astype(int)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build eval DataFrame: features + target in one frame (mlflow.evaluate expects this)
eval_df = X_test.copy()
eval_df[LABEL_COL] = y_test.values

logger.info(f"Eval set: {len(eval_df):,} rows | Fraud rate: {y_test.mean():.4f}")

# COMMAND ----------

# DBTITLE 1,Resolve champion and challenger models
# ── Get both champion and challenger pyfunc versions ──────────────────
def _get_version_by_alias(model_name: str, alias: str):
    """Return ModelVersion for the given alias, or None."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except mlflow.exceptions.MlflowException:
        return None


challenger_mv = _get_version_by_alias(_pyfunc_model, "challenger")
if challenger_mv is None:
    raise RuntimeError("No @challenger alias found on pyfunc model. Run pyfunc_wrapper.py first.")

champion_mv = _get_version_by_alias(_pyfunc_model, "champion")
if champion_mv is None:
    raise RuntimeError("No @champion alias found on pyfunc model.")

# Map to sklearn versions: latest = challenger, previous = champion
sklearn_all = sorted(
    client.search_model_versions(f"name='{_sklearn_model}'"),
    key=lambda v: int(v.version), reverse=True,
)
challenger_sklearn = sklearn_all[0]
champion_sklearn = sklearn_all[1] if len(sklearn_all) > 1 else sklearn_all[0]

challenger_uri = f"models:/{_sklearn_model}/{challenger_sklearn.version}"
champion_uri = f"models:/{_sklearn_model}/{champion_sklearn.version}"

logger.info(f"@champion  pyfunc v{champion_mv.version}  → sklearn v{champion_sklearn.version}")
logger.info(f"@challenger pyfunc v{challenger_mv.version} → sklearn v{challenger_sklearn.version}")

# COMMAND ----------

# DBTITLE 1,MLflow Evaluate — classification report
# MAGIC %md
# MAGIC ## MLflow Evaluate
# MAGIC Runs `mlflow.evaluate()` with `model_type="classifier"`. This auto-generates:
# MAGIC - **Metrics**: accuracy, precision, recall, F1, ROC-AUC, log loss (per-class and weighted)
# MAGIC - **Artifacts**: confusion matrix, ROC curve, precision-recall curve, lift chart
# MAGIC - **Evaluation tab**: compare any two runs side-by-side in the MLflow UI

# COMMAND ----------

# DBTITLE 1,Evaluate champion and challenger
import mlflow.sklearn
from sklearn.preprocessing import FunctionTransformer


def _load_and_patch(uri):
    """Load sklearn model and patch 'passthrough' for sklearn 1.3→1.7 compat."""
    m = mlflow.sklearn.load_model(uri)
    pp = m.named_steps.get("preprocessor")
    if pp is not None:
        for i, (name, trans, cols) in enumerate(pp.transformers_):
            if isinstance(trans, str) and trans == "passthrough":
                pp.transformers_[i] = (name, FunctionTransformer(), cols)
    return m


def _evaluate(uri, run_name, alias, pyfunc_ver, sklearn_ver):
    """Load model, predict, run mlflow.evaluate(), return (run, result, preds_df)."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("evaluation_type", "mlflow_evaluate")
        mlflow.set_tag("model_alias", alias)
        mlflow.set_tag("pyfunc_version", pyfunc_ver)
        mlflow.set_tag("sklearn_version", sklearn_ver)
        mlflow.set_tag("sklearn_model", _sklearn_model)

        model = _load_and_patch(uri)
        preds_df = eval_df.copy()
        preds_df["prediction"] = model.predict(eval_df[FEATURE_COLS])

        result = mlflow.evaluate(
            data=preds_df,
            predictions="prediction",
            targets=LABEL_COL,
            model_type="classifier",
        )
    logger.info(f"{alias} evaluation → run {run.info.run_id[:8]}...")
    return run, result, preds_df


# ── Evaluate both models ─────────────────────────────────────────────
champion_run, champion_result, champion_preds = _evaluate(
    champion_uri,
    f"champion_v{champion_mv.version}_evaluation",
    "champion", champion_mv.version, champion_sklearn.version,
)

challenger_run, challenger_result, challenger_preds = _evaluate(
    challenger_uri,
    f"challenger_v{challenger_mv.version}_evaluation",
    "challenger", challenger_mv.version, challenger_sklearn.version,
)

logger.info(f"\nCompare in MLflow UI → select both runs → Evaluation tab")

# COMMAND ----------

# DBTITLE 1,Display champion vs challenger metrics
# ── Side-by-side comparison table ────────────────────────────────────
champ_m = champion_result.metrics
chall_m = challenger_result.metrics

comparison = pd.DataFrame([
    {"metric": k,
     "champion": round(champ_m.get(k, 0.0), 4),
     "challenger": round(chall_m.get(k, 0.0), 4)}
    for k in sorted(set(champ_m) | set(chall_m))
])
logger.info("\nChampion vs Challenger (mlflow.evaluate metrics):")
display(comparison)

logger.info(f"\nArtifacts per run: {list(challenger_result.artifacts.keys())}")

# COMMAND ----------

# DBTITLE 1,Quality gate
# MAGIC %md
# MAGIC ## Quality Gate
# MAGIC Same minimum thresholds as `evaluate.py` — fail the pipeline if challenger is below acceptable quality.

# COMMAND ----------

# DBTITLE 1,Quality gate check
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1_score


def quality_gate(eval_result, eval_data, min_recall=0.30, min_precision=0.40, min_f1=0.35):
    """Quality gate on FRAUD CLASS (class=1) metrics — not weighted averages.

    Weighted averages mask poor fraud-class performance because the majority
    class (legit, 94.5%) dominates. For a fraud model, the gate must check
    how well we detect actual fraud.

    Also logs weighted averages from mlflow.evaluate() for reference.
    """
    # ── Fraud-class metrics (what actually matters) ───────────────────
    predictions = eval_data["prediction"].values
    actuals = eval_data[LABEL_COL].values

    fraud_recall = recall_score(actuals, predictions, pos_label=1)
    fraud_precision = precision_score(actuals, predictions, pos_label=1)
    fraud_f1 = sk_f1_score(actuals, predictions, pos_label=1)

    # ── Weighted averages from mlflow.evaluate (for reference) ───────
    weighted_metrics = eval_result.metrics

    logger.info(f"\n@challenger (pyfunc v{challenger_mv.version}) quality gate:")
    logger.info(f"")
    logger.info(f"  FRAUD CLASS (pos_label=1) — used for gating:")
    logger.info(f"    recall:    {fraud_recall:.4f}  (min: {min_recall})")
    logger.info(f"    precision: {fraud_precision:.4f}  (min: {min_precision})")
    logger.info(f"    f1_score:  {fraud_f1:.4f}  (min: {min_f1})")
    logger.info(f"")
    logger.info(f"  WEIGHTED AVG (reference only — dominated by majority class):")
    logger.info(f"    recall:    {weighted_metrics.get('recall_score', 0.0):.4f}")
    logger.info(f"    precision: {weighted_metrics.get('precision_score', 0.0):.4f}")
    logger.info(f"    f1_score:  {weighted_metrics.get('f1_score', 0.0):.4f}")

    failures = []
    if fraud_recall < min_recall:
        failures.append(f"fraud recall {fraud_recall:.4f} < {min_recall}")
    if fraud_precision < min_precision:
        failures.append(f"fraud precision {fraud_precision:.4f} < {min_precision}")
    if fraud_f1 < min_f1:
        failures.append(f"fraud f1_score {fraud_f1:.4f} < {min_f1}")

    if failures:
        msg = f"Challenger v{challenger_mv.version} REJECTED: {'; '.join(failures)}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"\nChallenger v{challenger_mv.version} PASSED quality gate — ready for A/B deployment.")
    logger.info(f"View full evaluation in MLflow UI: Experiment > select both runs > Evaluation tab")

# COMMAND ----------

# DBTITLE 1,Run
# Runs automatically as a DABs job task.
# Gates the challenger before 50/50 deployment.
# View confusion matrix, ROC, PR curve in MLflow UI > Evaluation tab.
quality_gate(challenger_result, challenger_preds, min_recall=0.30, min_precision=0.40, min_f1=0.35)
-- =============================================================
-- SQL Alert Definitions for Fraud Model Drift Monitoring
-- Create these as Databricks SQL Alerts pointing at
-- drift metrics table: ml.fraud.drift_metrics
-- =============================================================

-- Alert 1: Feature PSI exceeds threshold
-- Schedule: daily | Trigger: rows > 0
SELECT
  run_date,
  CASE
    WHEN txn_count_1d_psi     > 0.2 THEN 'txn_count_1d'
    WHEN txn_count_7d_psi     > 0.2 THEN 'txn_count_7d'
    WHEN txn_amount_avg_1d_psi > 0.2 THEN 'txn_amount_avg_1d'
    WHEN txn_amount_avg_7d_psi > 0.2 THEN 'txn_amount_avg_7d'
    WHEN amount_zscore_psi    > 0.2 THEN 'amount_zscore'
    WHEN amount_psi           > 0.2 THEN 'amount'
  END AS drifted_feature,
  GREATEST(
    COALESCE(txn_count_1d_psi, 0), COALESCE(txn_count_7d_psi, 0),
    COALESCE(txn_amount_avg_1d_psi, 0), COALESCE(txn_amount_avg_7d_psi, 0),
    COALESCE(amount_zscore_psi, 0), COALESCE(amount_psi, 0)
  ) AS max_psi
FROM ml.fraud.drift_metrics
WHERE run_date >= current_date() - INTERVAL 1 DAY
  AND GREATEST(
    COALESCE(txn_count_1d_psi, 0), COALESCE(txn_count_7d_psi, 0),
    COALESCE(txn_amount_avg_1d_psi, 0), COALESCE(txn_amount_avg_7d_psi, 0),
    COALESCE(amount_zscore_psi, 0), COALESCE(amount_psi, 0)
  ) > 0.2;


-- Alert 2: Prediction volume drop (pipeline failure)
-- Schedule: hourly | Trigger: rows > 0
SELECT
  current_timestamp() AS check_time,
  COUNT(*) AS prediction_count_last_hour
FROM ml.fraud.fraud_predictions
WHERE prediction_ts >= current_timestamp() - INTERVAL 1 HOUR
HAVING COUNT(*) < 100;


-- Alert 3: Fraud rate anomaly
-- Schedule: daily | Trigger: rows > 0
SELECT
  DATE(prediction_ts) AS prediction_date,
  AVG(CAST(predicted_label AS DOUBLE)) AS daily_fraud_rate
FROM ml.fraud.fraud_predictions
WHERE prediction_ts >= current_date() - INTERVAL 1 DAY
GROUP BY DATE(prediction_ts)
HAVING daily_fraud_rate > 0.10 OR daily_fraud_rate < 0.001;

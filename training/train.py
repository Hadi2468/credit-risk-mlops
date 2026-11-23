# training/train.py
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURES = [
    "num_payments", "avg_payment_delay", "max_payment_delay",
    "num_missed_payments", "num_late_payments", "age", "monthly_income",
]
MODEL_NAME = "credit_risk"

# PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}    # Run 1
# PARAMS = {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05}    # Run 2
# PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}    # Run 3
# PARAMS = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.01}    # Run 4
# PARAMS = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.01}    # Run 5
# PARAMS = {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.01}    # Run 6
PARAMS = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.01}    # Run 7

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit_risk")

# 1 READ
df = pd.read_parquet("data/training/training_set.parquet")
X, y = df[FEATURES], df["defaulted"]

# 2 SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

with mlflow.start_run() as run:
    # 3 TRAIN
    model = XGBClassifier(**PARAMS, eval_metric="auc")
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # 4 LOG
    mlflow.log_params(PARAMS)
    mlflow.log_param("train_data", "data/training/training_set.parquet")
    mlflow.log_param("n_rows", len(df))
    mlflow.log_metric("auc", auc)
    info = mlflow.xgboost.log_model(
        model,
        name="model",
        input_example=X_test.head(3),
        registered_model_name=MODEL_NAME,
    )
    print(f"run_id: {run.info.run_id}   auc: {auc:.3f}")

# promote only if better than the current champion
client = MlflowClient()
version = info.registered_model_version

try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_auc = client.get_run(champion.run_id).data.metrics["auc"]
except Exception:
    champion = None
    champion_auc = -1.0

if auc > champion_auc:
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"PROMOTED: version {version} (auc {auc:.3f}) beats champion (auc {champion_auc:.3f})")
else:
    print(f"KEPT: champion is still version {champion.version} (auc {champion_auc:.3f}); new version {version} has auc {auc:.3f}")
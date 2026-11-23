# app/main.py
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from mlflow import MlflowClient
from pydantic import BaseModel, Field

FEATURES = [
    "num_payments", "avg_payment_delay", "max_payment_delay",
    "num_missed_payments", "num_late_payments", "age", "monthly_income",
]
MODEL_NAME = "credit_risk"

# load once, at startup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
model_version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
store = FeatureStore(repo_path="feature_repo")

app = FastAPI(title="Credit Risk API")


class PredictRequest(BaseModel):
    user_id: int = Field(ge=1)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_version}


@app.post("/predict")
def predict(req: PredictRequest):
    # 1 features from the Feast online store
    features = store.get_online_features(
        features=[f"user_payment_features:{f}" for f in FEATURES],
        entity_rows=[{"user_id": req.user_id}],
    ).to_dict()

    # 2 unknown user: Feast returns None
    if features["num_payments"][0] is None:
        raise HTTPException(status_code=404, detail="user not found")

    # 3 same column order as in training
    X = pd.DataFrame({f: features[f] for f in FEATURES})

    # 4 score with the champion model
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "user_id": req.user_id,
        "default_probability": round(probability, 3),
        "model_version": model_version,
    }
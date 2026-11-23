# Credit Risk MLOps

### An end-to-end MLOps pipeline for real-time credit-risk scoring: one feature definition from training to serving

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Feast](https://img.shields.io/badge/Feast-feature%20store-orange)
![MLflow](https://img.shields.io/badge/MLflow-tracking%20%26%20registry-0194E2)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688)

A credit-risk model is only as good as the infrastructure around it. This project builds that infrastructure:
**one feature definition** shared by training and serving, **point-in-time correct** training data,
**tracked and versioned** models with a promotion gate, and a **low-latency** prediction API.

---

## 🧩 Three problems, three answers

| Problem | What goes wrong | Answer in this repo |
|---|---|---|
| Training-serving skew | the same feature is computed twice, in two codebases, with different results | Feast: features defined once, read by training (offline) and serving (online) |
| "Which model is live?" | model files passed around by hand, no record of data or params | MLflow: every run tracked, API loads `models:/credit_risk@champion` |
| Data leakage | training rows see payments that happened after the loan application | PySpark snapshots + Feast point-in-time join |

---

## ✨ Highlights

| Capability | Detail |
|---|---|
| Feature engineering | PySpark builds monthly snapshots; each one aggregates only payments **before** its date |
| Feature store | Feast with a file offline store and a SQLite online store, 60-day TTL |
| No skew, verified | offline snapshot and online store return identical values for the same user |
| Point-in-time training set | Feast joins each loan to the latest snapshot at or before its application date |
| Experiment tracking | XGBoost runs logged to MLflow: params, AUC, training data path |
| Promotion gate | `champion` alias moves only if the new version beats the current champion |
| Real-time serving | FastAPI takes only `user_id`; features from Feast, model from MLflow |
| Input safety | Pydantic validation (`422`), unknown users (`404`) |

---

## 🔀 Architecture

```
synthetic loan data
        |
PySpark feature snapshots
        |
Feast offline store ----(point-in-time join)----> training set
        |                                              |
  feast materialize                          XGBoost + MLflow
        |                                    (tracking, registry, promotion gate)
        v                                              |
Feast online store (SQLite) <--- user_id --- FastAPI /predict <--- @champion
```

---

## 🚀 Quickstart

```bash
conda create -n mlops python=3.12 -y
conda activate mlops
pip install -r requirements.txt
```

### 1. Data and features

```bash
python data/generate_data.py
python features/build_features.py
```

### 2. Feature store

```bash
cd feature_repo
feast apply
feast materialize 2026-01-01T00:00:00 2026-11-22T00:00:00
cd ..
```

### 3. Training

```bash
python scripts/build_training_set.py
python training/train.py
```

### 4. REST API

```bash
uvicorn app.main:app
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 2}'
```

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | status and live model version |
| `/predict` | POST | `default_probability` and `model_version` |

---

## 📊 Results

| Metric | Value |
|---|---|
| Feast online lookup (in-process) | ~0.2 ms |
| `/predict` end-to-end, 200 HTTP requests | p50 6.7 ms, p95 17.4 ms, p99 23.6 ms |
| Model AUC | ~0.62 (synthetic labels are random draws; first run 0.56 was overfitting, fixed with depth 2) |

---

## 🔭 Observability

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Every training run with its params and AUC, every registered version, and which one is `@champion`.

---

## 🧪 Testing

```bash
python -m pytest -v
```

| Test | Checks |
|---|---|
| `test_health` | `200` |
| `test_predict_known_user` | `200`, probability between 0 and 1, model version present |
| `test_predict_unknown_user` | `404` |
| `test_predict_invalid_user_id` | `422` |

---

## 🗂️ Project structure

```
credit-risk-mlops/
├── app/main.py                     FastAPI service
├── data/generate_data.py           synthetic users, payments, loans
├── features/build_features.py      PySpark point-in-time snapshots
├── feature_repo/                   Feast definitions and settings
├── scripts/
│   ├── build_training_set.py       point-in-time join
│   ├── read_online.py              online lookup check
│   └── benchmark_latency.py        p50 / p95 / p99
├── training/train.py               XGBoost + MLflow + promotion gate
└── tests/test_api.py               API tests
```

---

## ⚙️ Configuration

| Setting | Local | Production |
|---|---|---|
| Online store | SQLite | Redis or DynamoDB |
| Offline store | parquet files | warehouse (e.g. Databricks / Delta tables) |
| MLflow backend | `sqlite:///mlflow.db` | managed MLflow server |
| Feature TTL | 60 days | per feature view |

---

## ⚠️ Disclaimer

Synthetic data, built to demonstrate MLOps infrastructure. Not a real credit model and not for lending decisions.

---

## 🛣️ Roadmap

→ Drift monitoring with Evidently (data and prediction drift on logged requests)
→ Orchestrate the pipeline as Dagster assets
→ Skew test in CI
→ Faster model input (NumPy instead of pandas)
→ Docker image

---

## 📚 Data

Fully synthetic: 2,000 users, ~15,000 payments (with missed payments as `NULL`), 2,000 loan applications.
Each user has a hidden risk level that drives payment delays and defaults; it is not saved, so the model must learn it from behavior.

---

## 🧑🏻‍💻 Author

**Hadi Hosseini**  
AI/ML Engineer  
[LinkedIn](https://www.linkedin.com/in/hadi468)
# scripts/build_training_set.py
import os
import pandas as pd

from feast import FeatureStore

FEATURES = [
    "user_payment_features:num_payments",
    "user_payment_features:avg_payment_delay",
    "user_payment_features:max_payment_delay",
    "user_payment_features:num_missed_payments",
    "user_payment_features:num_late_payments",
    "user_payment_features:age",
    "user_payment_features:monthly_income",
]

# who + when: one row per loan, at the application date
loans = pd.read_parquet("data/raw/loans.parquet")
entity_df = loans.rename(columns={"application_date": "event_timestamp"})[
    ["loan_id", "user_id", "event_timestamp", "loan_amount", "defaulted"]
]

# point-in-time join
store = FeatureStore(repo_path="feature_repo")
training = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()

print("rows:", len(training))
print("nulls per column:")
print(training.isna().sum())
print(training.sort_values("loan_id").head())

os.makedirs("data/training", exist_ok=True)
training.to_parquet("data/training/training_set.parquet", index=False)
print("saved data/training/training_set.parquet")
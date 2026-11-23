# scripts/read_online.py
import time

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

store = FeatureStore(repo_path="feature_repo")


# one user (user_id = 2)
result = store.get_online_features(
    features=FEATURES,
    entity_rows=[{"user_id": 2}],
).to_dict()
print(result)


# two users at once (user_ids = 1 & 2)
result = store.get_online_features(
    features=FEATURES,
    entity_rows=[{"user_id": 1}, {"user_id": 2}],
).to_dict()
print(result)


# a user who does not exist (user_id = 99999)
result = store.get_online_features(
    features=FEATURES,
    entity_rows=[{"user_id": 99999}],
).to_dict()
print(result)


# how fast?
start = time.perf_counter()
for _ in range(100):
    store.get_online_features(features=FEATURES, entity_rows=[{"user_id": 2}])
elapsed_ms = (time.perf_counter() - start) * 1000 / 100
print(f"average online lookup: {elapsed_ms:.1f} ms")
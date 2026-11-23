# feature_repo/definitions.py
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

FEATURES_FILE = Path(__file__).resolve().parent.parent / "data" / "features" / "user_features.parquet"


# 1 ENTITY:
# ------------------------------------------------------------
user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.INT64)


# 2 SOURCE:
# ------------------------------------------------------------
user_features_source = FileSource(
    path=str(FEATURES_FILE),
    timestamp_field="event_timestamp",
)


# 3 FEATURE VIEW:
# ------------------------------------------------------------
user_payment_features = FeatureView(
    name="user_payment_features",
    entities=[user],
    ttl=timedelta(days=60),
    schema=[
        Field(name="num_payments", dtype=Int64),
        Field(name="avg_payment_delay", dtype=Float64),
        Field(name="max_payment_delay", dtype=Float64),
        Field(name="num_missed_payments", dtype=Int64),
        Field(name="num_late_payments", dtype=Int64),
        Field(name="age", dtype=Int64),
        Field(name="monthly_income", dtype=Float64),
    ],
    source=user_features_source,
)
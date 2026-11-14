import os, sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# make Spark workers use the same Python as this script (the mlops env)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (
    SparkSession.builder.appName("build_features")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

# 1 READ
payments = spark.read.parquet("data/raw/payments.parquet")
users = spark.read.parquet("data/raw/users.parquet")

# 2 GRID: one row per (user, snapshot date)
snapshot_dates = [
    "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01",
    "2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01",
]
snapshots = spark.createDataFrame([(d,) for d in snapshot_dates], ["event_timestamp"])
snapshots = snapshots.withColumn("event_timestamp", F.to_timestamp("event_timestamp"))
g = users.select("user_id").crossJoin(snapshots).alias("g")
p = payments.alias("p")

# 3 JOIN: each snapshot sees only payments BEFORE its date
joined = g.join(
    p,
    (F.col("g.user_id") == F.col("p.user_id"))
    & (F.col("p.payment_date") < F.col("g.event_timestamp")),
    "left",
)

# 4 AGGREGATE: one row per user per snapshot
is_missed = F.col("p.payment_date").isNotNull() & F.col("p.delay_days").isNull()
is_late = F.col("p.delay_days") > 7

features = joined.groupBy("g.user_id", "g.event_timestamp").agg(
    F.count("p.payment_date").alias("num_payments"),
    F.avg("p.delay_days").alias("avg_payment_delay"),
    F.max("p.delay_days").alias("max_payment_delay"),
    F.sum(F.when(is_missed, 1).otherwise(0)).alias("num_missed_payments"),
    F.sum(F.when(is_late, 1).otherwise(0)).alias("num_late_payments"),
)

# 5 ADD user columns
features = features.join(users, "user_id")

# 6 WRITE: Spark computes, pandas writes (no winutils on this machine)
features.orderBy("user_id", "event_timestamp").show(8)
print("rows:", features.count())

os.makedirs("data/features", exist_ok=True)
features.toPandas().to_parquet("data/features/user_features.parquet", index=False)
print("saved data/features/user_features.parquet")

spark.stop()
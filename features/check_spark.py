from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("check").master("local[*]").getOrCreate()

df = spark.read.parquet("data/raw/payments.parquet")
df.show(5)
print("rows:", df.count())

spark.stop()
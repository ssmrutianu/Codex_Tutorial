"""
Run a local Spark job that translates the sample CSV and prints row count.
"""

from pyspark.sql import SparkSession
from text_translation import add_translation_column, row_count

spark = SparkSession.builder.appName("codex-translate-demo").getOrCreate()

df = spark.read.csv("codex_multilang_samples.csv", header=True)
df = add_translation_column(df)
df.show(truncate=False)

print("⚠️  Row count (buggy):", row_count(df))
spark.stop()

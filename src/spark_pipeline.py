from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col, when

spark = (
    SparkSession.builder
    .appName("HealthcareBigDataProject")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("data/synthetic/insurance_large.csv", header=True, inferSchema=True)

print("\nSchema:")
df.printSchema()

print("\nSample rows:")
df.show(5, truncate=False)

print("\nTotal rows:")
print(df.count())

print("\nAverage charges by smoker:")
df.groupBy("smoker").agg(
    avg("charges").alias("avg_charges"),
    avg("hospital_visits").alias("avg_hospital_visits"),
    count("patient_id").alias("patient_count")
).show()

df = df.withColumn(
    "risk_segment",
    when(col("high_cost") == 1, "high")
    .when(col("charges") > 10000, "medium")
    .otherwise("low")
)

print("\nRisk segments:")
df.groupBy("risk_segment").count().show()

spark.stop()
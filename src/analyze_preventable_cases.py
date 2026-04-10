from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col, round as spark_round

spark = (
    SparkSession.builder
    .appName("PreventableCaseAnalysis")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("data/synthetic/insurance_large.csv", header=True, inferSchema=True)

print("\nTotal patients:")
print(df.count())

print("\nPreventable case count:")
df.groupBy("preventable_case").count().show()

print("\nAverage charges by preventable case:")
df.groupBy("preventable_case").agg(
    spark_round(avg("charges"), 2).alias("avg_charges"),
    spark_round(avg("hospital_visits"), 2).alias("avg_hospital_visits"),
    spark_round(avg("primary_care_visits"), 2).alias("avg_primary_care_visits")
).show()

print("\nPreventable cases by smoker:")
df.filter(col("preventable_case") == 1).groupBy("smoker").agg(
    count("*").alias("count"),
    spark_round(avg("charges"), 2).alias("avg_charges")
).show()

print("\nPreventable cases by region:")
df.filter(col("preventable_case") == 1).groupBy("region").agg(
    count("*").alias("count"),
    spark_round(avg("charges"), 2).alias("avg_charges")
).show()

spark.stop()
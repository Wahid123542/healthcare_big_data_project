from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = (
    SparkSession.builder
    .appName("ModelInsights")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("data/synthetic/insurance_large.csv", header=True, inferSchema=True)

categorical_cols = ["sex", "smoker", "region"]
numeric_cols = [
    "age",
    "bmi",
    "children",
    "primary_care_visits",
    "emergency_visits",
    "hospital_visits",
    "preventive_visit_flag",
    "chronic_condition_score",
    "medication_count",
]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=numeric_cols + [f"{c}_ohe" for c in categorical_cols],
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="high_cost",
    maxIter=20
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

lr_model = model.stages[-1]

print("\nIntercept:")
print(lr_model.intercept)

print("\nTop coefficients:")
coeffs = lr_model.coefficients.toArray()

for i, coef in enumerate(coeffs):
    print(f"Feature {i}: {coef:.4f}")

spark.stop()
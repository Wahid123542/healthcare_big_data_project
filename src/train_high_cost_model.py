from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("HighCostPrediction")
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
preds = model.transform(test_df)

print("\nPredictions sample:")
preds.select("high_cost", "prediction", "probability").show(5, truncate=False)

auc_eval = BinaryClassificationEvaluator(labelCol="high_cost", rawPredictionCol="rawPrediction")
auc = auc_eval.evaluate(preds)

acc_eval = MulticlassClassificationEvaluator(labelCol="high_cost", predictionCol="prediction", metricName="accuracy")
f1_eval = MulticlassClassificationEvaluator(labelCol="high_cost", predictionCol="prediction", metricName="f1")

accuracy = acc_eval.evaluate(preds)
f1 = f1_eval.evaluate(preds)

print(f"\nAUC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion matrix counts:")
preds.groupBy("high_cost", "prediction").count().orderBy("high_cost", "prediction").show()

spark.stop()
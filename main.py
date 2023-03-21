import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.master("local[*]").getOrCreate()

input_train_df = spark.read.csv("./InputTrain.csv",header=True,inferSchema=True)
labels_df = spark.read.csv("./StepOne_LabelTrain.csv",header=True,inferSchema=True)

merged_df = input_train_df.join(labels_df, on="Index",how="inner")

# Define the feature columns and the target column
feature_cols = input_train_df.columns[2:]  # the first two columns are id and house id
target_col = "Washing Machine"
label_cols = ["Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle"]

# Assemble the features into a vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(merged_df).select(col("features"), *label_cols)

# Split the data into training and validation sets
(training_data, validation_data) = data.randomSplit([0.8, 0.2], seed=42)

# Train the GBT model
gbt = GBTClassifier(labelCol=target_col, featuresCol="features", maxIter=10)
model = gbt.fit(training_data)

# Evaluate the model on the validation set
evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
predictions = model.transform(validation_data)
accuracy = evaluator.evaluate(predictions)
print("Accuracy on validation set = %g" % accuracy)

# Use the model to predict the state of the 5 appliances on the test data
test_data = spark.read.csv("InputTest.csv", header=True, inferSchema=True)
test_features = assembler.transform(test_data).select(col("features"))
test_features.printSchema()
test_predictions = model.transform(test_features).select(col("prediction"), col("Washing Machine"), col("Dishwasher"), col("Tumble Dryer"), col("Microwave"), col("Kettle"))

# Save the predictions in the format specified in the instructions
test_predictions.write.csv("StepOne_Predictions.csv", header=True)
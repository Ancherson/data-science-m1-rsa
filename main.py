from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "6g").getOrCreate()

# Load the training data
train_data = spark.read.csv("InputTrain.csv", header=True, inferSchema=True)

# Load the labels for the training data
train_labels = spark.read.csv("StepOne_LabelTrain.csv", header=True, inferSchema=True)

# Join the training data with the labels
train_data = train_data.join(train_labels, "Index")

# Extract the features from the training data
feature_cols = [c for c in train_data.columns if c not in ["Index", "House_id", "Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle"]]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data = assembler.transform(train_data).select("features", "Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle")

# Train a GBT classifier for each appliance
models = {}
evaluator = MulticlassClassificationEvaluator(metricName="weightedPrecision")
for appliance in ["Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle"]:
    train_data = train_data.withColumnRenamed(f"{appliance}_consumption", appliance)
    gbt = GBTClassifier(labelCol=appliance)
    model = gbt.fit(train_data)
    models[appliance] = model

    # Evaluate the model on the training data
    predictions = model.transform(train_data)
    predictions = predictions.withColumn("label", col(appliance))
    precision = evaluator.evaluate(predictions)
    print(f"{appliance} model precision: {precision}")

# Load the test data
test_data = spark.read.csv("InputTest.csv", header=True, inferSchema=True)

# Extract the features from the test data
test_data = assembler.transform(test_data).select("Index", "House_id", "features")

# Make predictions for each appliance and generate the binary vectors
binary_vectors = []
for row in test_data.collect():
    index = row.Index
    house_id = row.House_id
    features = row.features
    feature_array = features.toArray() # Convert to NumPy array
    feature_list = feature_array.tolist()
    feature_df = spark.createDataFrame([(Vectors.dense(feature_array),)], ["features"])
    binary_vector = [index, house_id]
    for appliance in ["Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle"]:
        model = models[appliance]
        prediction = model.transform(feature_df).select("prediction").collect()[0][0]
        binary_vector.append(int(prediction))
    binary_vectors.append(binary_vector)

# Save the binary vectors to a CSV file
binary_vectors_df = spark.createDataFrame(binary_vectors, ["Index", "House_id", "Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle"])
binary_vectors_df.drop("House_id")
binary_vectors_df.write.csv("StepOne_Predictions.csv", header=True, mode="overwrite")
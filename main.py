import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()

df_x = pd.read_csv("./InputTrain.csv")
df_y = pd.read_csv("./StepOne_LabelTrain.csv")
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()

df_x = pd.read_csv("./InputTrain.csv")
df_y = pd.read_csv("./StepOne_LabelTrain.csv")
merged = pd.merge(df_x.drop(columns=["House_id"]),df_y.drop(columns=["House_id"]),on="Index",how="inner")
merged = merged.drop(columns = ["Index"])

prossecing = merged.groupby(["Washing Machine","Dishwasher","Tumble Dryer","Microwave","Kettle"])
print(prossecing.describe())
print(prossecing.max())
print(prossecing.min())
print(prossecing.mean())
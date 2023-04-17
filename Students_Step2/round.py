import pandas
import numpy


data_1 = pandas.read_csv("results (1).csv")
data_2 = pandas.read_csv("results.csv")


wash = numpy.minimum(data_1["Washing Machine"].values,data_2["Washing Machine"].values)
Dish = numpy.minimum(data_1["Dishwasher"].values,data_2["Dishwasher"].values)
tumbl = numpy.minimum(data_1["Tumble Dryer"].values,data_2["Tumble Dryer"].values)
micro = numpy.minimum(data_1["Microwave"].values,data_2["Microwave"].values)
kett = numpy.minimum(data_1["Kettle"].values,data_2["Kettle"].values)

all = pandas.DataFrame()
all["Washing Machine"] = wash
all["Dishwasher"] = Dish
all["Tumble Dryer"] = tumbl
all["Microwave"] = micro
all["Kettle"] = kett



all.to_csv('mixed.csv', index=True,index_label="Index")
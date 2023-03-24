import pandas as pd
from sklearn.linear_model import LogisticRegression

# load data
consommation1_df = pd.read_csv("InputTrain.csv")
consommation2_df = pd.read_csv("InputTest.csv")
activation_df = pd.read_csv("StepOne_LabelTrain.csv")

# select columns corresponding to the appliances in the activation file
appareils = ['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']
x = consommation1_df.values

# initialize a dictionary to store the classification results
resultats = ({appareil: [] for appareil in appareils})

# loop over each appliance and train a classifier to predict its activation
for appareil in appareils:
    print(appareil)
    y = activation_df[appareil].values
    lr = LogisticRegression(random_state=0)
    lr.fit(x, y)
    resultats[appareil] = lr.predict(consommation2_df)

# convert the results to a pandas DataFrame and save to a CSV file
print("saving results")
resultats_df = pd.DataFrame(resultats)
resultats_df.to_csv("lrResults.csv", index=True, index_label='Index', columns=appareils)
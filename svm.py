import pandas as pd
from sklearn.svm import SVC

# load data
consommation1_df = pd.read_csv("InputTrain.csv")
consommation2_df = pd.read_csv("InputTest.csv")
activation_df = pd.read_csv("StepOne_LabelTrain.csv")

# select columns corresponding to appliances in the activation file
appareils = ['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']
x = consommation1_df.values

# initialize a dictionary to store the classification results
resultats = ({appareil: [] for appareil in appareils})

# loop over each appliance and train a classifier to predict its activation
for appareil in appareils:
    print(appareil)
    y = activation_df[appareil].values
    clf = SVC(kernel='linear', random_state=0) #kernel=sigmoid > poly > rbf, linear too long
    clf.fit(x, y)
    resultats[appareil] = clf.predict(consommation2_df)

# convert the results to a pandas DataFrame and save to a CSV file
print("saving results")
resultats_df = pd.DataFrame(resultats)
resultats_df.to_csv("svmResultsLinear.csv", index=True, index_label='Index', columns=appareils)
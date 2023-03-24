import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

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
    et = ExtraTreesClassifier(random_state=0)
    param_grid = {
        'n_estimators': [100, 250, 500, 750, 1000, 1500]  # specify values to search
    }
    grid_search = GridSearchCV(et, param_grid, cv=5)
    grid_search.fit(x, y)
    print(f"Best parameters for {appareil}: {grid_search.best_params_}")
    resultats[appareil] = grid_search.predict(consommation2_df)

# convert the results to a pandas DataFrame and save to a CSV file
print("saving results")
resultats_df = pd.DataFrame(resultats)
resultats_df.to_csv("etResultsGridSearch.csv", index=True, index_label='Index', columns=appareils)
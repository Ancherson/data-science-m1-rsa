import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Charger les fichiers CSV contenant les données de consommation et d'activation
consommation1_df = pd.read_csv("InputTrain.csv")
consommation2_df = pd.read_csv("InputTest.csv")
activation_df = pd.read_csv("StepOne_LabelTrain.csv")

# Sélectionner les colonnes correspondant aux appareils dans le fichier d'activation
appareils = ['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']
X = consommation1_df.values

# Initialiser un dictionnaire pour stocker les résultats de la classification
resultats = ({appareil: [] for appareil in appareils})


# Boucler sur chaque appareil et entraîner un classificateur binaire pour prédire son activation
for appareil in appareils:
    y = activation_df[appareil].values
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    resultats[appareil] = clf.predict(consommation2_df)
resultats['House_id'] = consommation2_df['House_id'].values.tolist()
for elt in resultats :
    print(elt)
    print(len(resultats[elt]))
# Convertir les résultats en un DataFrame pandas et les enregistrer dans un fichier CSV
resultats_df = pd.DataFrame(resultats)
resultats_df.to_csv("resultats.csv", index=True,index_label='Index',columns=['House_id','Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle'])
import pandas as pd
from sklearn.svm import SVC

# Charger les fichiers CSV contenant les données de consommation et d'activation
consommation1_df = pd.read_csv("InputTrain.csv")
consommation2_df = pd.read_csv("InputTest.csv")
activation_df = pd.read_csv("StepOne_LabelTrain.csv")

# Sélectionner les colonnes correspondant aux appareils dans le fichier d'activation
appareils = ['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']
x = consommation1_df.values

# Initialiser un dictionnaire pour stocker les résultats de la classification
resultats = ({appareil: [] for appareil in appareils})

# Boucler sur chaque appareil et entraîner un classificateur binaire pour prédire son activation
for appareil in appareils:
    print(appareil)
    y = activation_df[appareil].values
    clf = SVC(kernel='linear', random_state=0) #kernel=sigmoid > poly > rbf, linear
    clf.fit(x, y)
    resultats[appareil] = clf.predict(consommation2_df)

# Convertir les résultats en un DataFrame pandas et les enregistrer dans un fichier CSV
print("saving results")
resultats_df = pd.DataFrame(resultats)
resultats_df.to_csv("svmResultsLinear.csv", index=True, index_label='Index', columns=appareils)
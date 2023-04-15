import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# Import des fichiers de données
consumption = pd.read_csv("InputTrain.csv")
washing_machine = pd.read_csv("StepTwo_LabelTrain_WashingMachine.csv")
dishwasher = pd.read_csv("StepTwo_LabelTrain_Dishwasher.csv")
tumble_dryer = pd.read_csv("StepTwo_LabelTrain_TumbleDryer.csv")
microwave = pd.read_csv("StepTwo_LabelTrain_Microwave.csv")
kettle = pd.read_csv("StepTwo_LabelTrain_Kettle.csv")
df_test = pd.read_csv("InputTest.csv")

# Normalisation des données de consommation
scaler = StandardScaler()
consumption_norm = scaler.fit_transform(consumption)

# Combinaison des données de consommation normalisées avec les données étiquetées pour chaque appareil
x_train = np.concatenate((consumption_norm, washing_machine.iloc[:, 1:], dishwasher.iloc[:, 1:], 
                           tumble_dryer.iloc[:, 1:], microwave.iloc[:, 1:], kettle.iloc[:, 1:]), axis=1)

# Création du modèle
model = Sequential()

# Ajout des couches de convolution et de pooling
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Ajout de la couche de sortie
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='sigmoid'))

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
labels_train = np.concatenate((washing_machine.iloc[:, 1:], dishwasher.iloc[:, 1:], 
                                tumble_dryer.iloc[:, 1:], microwave.iloc[:, 1:], kettle.iloc[:, 1:]), axis=1)
labels_train = np.argmax(labels_train, axis=1)

model.fit(x_train.reshape(x_train.shape[0], x_train.shape[1], 1), to_categorical(labels_train, num_classes=5), epochs=10, batch_size=128)

# Prédiction sur les données de test
consumption_test_norm = scaler.transform(df_test)
X_test = np.concatenate((consumption_test_norm, np.zeros((df_test.shape[0], 5))), axis=1)
y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))

# Export résultats
df_pred = pd.DataFrame(y_pred, columns=["Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle"])
df_pred.to_csv("output.csv", index=False)
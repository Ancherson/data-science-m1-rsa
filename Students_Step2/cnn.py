import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

consumption = pd.read_csv("InputTrain.csv")
washing_machine = pd.read_csv("StepTwo_LabelTrain_WashingMachine.csv")
dishwasher = pd.read_csv("StepTwo_LabelTrain_Dishwasher.csv")
tumble_dryer = pd.read_csv("StepTwo_LabelTrain_TumbleDryer.csv")
microwave = pd.read_csv("StepTwo_LabelTrain_Microwave.csv")
kettle = pd.read_csv("StepTwo_LabelTrain_Kettle.csv")


def transform(machine,name):
    time_stamps = []
    for elt in machine.columns :
        if str.startswith(elt,'TimeStep') :
            time_stamps.append(elt)
    machine = pd.melt(machine,id_vars=['Index','House_id'],value_vars= time_stamps, var_name='time',value_name = name)

    machine = pd.DataFrame(machine)
    times = []
    for elt in machine['time']:
        times.append(int(elt[9:]))
    machine['time'] = times
    machine = machine.sort_values(['Index','time'])
    return machine


washing_machine = transform(washing_machine,'Washing Machine')
dishwasher = transform(dishwasher,'Dishwasher')
tumble_dryer = transform(tumble_dryer,'Tumble Dryer')
microwave = transform(microwave,'Microwave')
kettle = transform(kettle,'Kettle')

machines = [washing_machine,dishwasher,tumble_dryer,microwave,kettle]

consumption = transform(consumption,'watt')
# display(consumption)


df_train = pd.concat([consumption,washing_machine['Washing Machine'],dishwasher['Dishwasher'],tumble_dryer['Tumble Dryer'],microwave['Microwave'],kettle['Kettle']],keys=['Index', 'House_id','time','watt','Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle', 'watt'],axis=1,join="inner")
# display(df_train)
print(len(df_train[df_train["Index"] == 1]))


# Chargement des données
#df

# Paramètres du modèle
batch_size = 64
seq_len = 2160
n_features = 1
n_targets = 5

# Création du générateur de données d'entraînement
def generate_data(df, batch_size):
    num_curves = len(df['Index'].unique())
    num_batches_per_epoch = num_curves // batch_size

    while True:
        # Mélange aléatoire des courbes
        shuffled_curves = np.random.permutation(df['Index'].unique())

        # Pour chaque batch
        for i in range(num_batches_per_epoch):
            # Sélection aléatoire de batch_size courbes
            batch_curves = shuffled_curves[i*batch_size:(i+1)*batch_size]

            # Initialisation des tenseurs d'entrée et de sortie du batch
            X_batch = np.zeros((batch_size, seq_len, n_features))
            y_batch = np.zeros((batch_size, seq_len, n_targets))

            # Remplissage des tenseurs d'entrée et de sortie du batch
            for j, curve in enumerate(batch_curves):
                curve_data = df[df['Index'] == curve]
                X_batch[j, :, 0] = curve_data['watt'].values
                y_batch[j,:,:] = curve_data[['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']].values

            yield (X_batch, y_batch)

# Création du modèle
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_len, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_targets, activation='sigmoid'))

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy')

# Entraînement du modèle
model.fit_generator(generate_data(df_train, batch_size), steps_per_epoch=len(df_train['Index'].unique()) // batch_size, epochs=5)

# Charger les données de test dans une DataFrame Pandas
df_test = pd.read_csv("InputTest.csv")
df_test = transform(df_test,"watt")

# Génération des données de test
def generate_test_data(df):
    for _, curve_data in df.groupby('Index'):
        X = np.array(curve_data[['watt']].values).reshape(1, -1, 1)
        yield X

# Prédiction des états des appareils pour chaque courbe de consommation
y_pred = []
for X_test in generate_test_data(df_test):
    y_test_pred = model.predict(X_test)
    y_pred.append(y_test_pred[0,:,:])

# Création d'un DataFrame pour les résultats
result_df = pd.DataFrame(data=np.array(y_pred).reshape(-1, 5), columns=['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle'])
result_df = result_df.round(4)

# Exportation des résultats au format CSV
result_df.to_csv('results.csv', index=True,index_label="Index")

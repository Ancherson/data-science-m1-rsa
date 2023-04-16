import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Reshape
from keras.optimizers import Adam

# load the data
consumption = pd.read_csv("InputTrain.csv")
washing_machine = pd.read_csv("StepTwo_LabelTrain_WashingMachine.csv")
dishwasher = pd.read_csv("StepTwo_LabelTrain_Dishwasher.csv")
tumble_dryer = pd.read_csv("StepTwo_LabelTrain_TumbleDryer.csv")
microwave = pd.read_csv("StepTwo_LabelTrain_Microwave.csv")
kettle = pd.read_csv("StepTwo_LabelTrain_Kettle.csv")

# transform the label data into a single column
def transform(machine,name):
    time_stamps = []
    for elt in machine.columns :
        if str.startswith(elt,'TimeStep') :
            time_stamps.append(elt)
    machine = pd.melt(machine,id_vars=['Index','House_id'], value_vars=time_stamps, var_name='time', value_name=name)

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

df_train = pd.concat([consumption,washing_machine['Washing Machine'],dishwasher['Dishwasher'],tumble_dryer['Tumble Dryer'],microwave['Microwave'],kettle['Kettle']],names=['Index', 'House_id','time','watt','Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle', 'watt'],axis=1,join="inner")
print(len(df_train[df_train["Index"] == 1]))

# model parameters
batch_size = 64
seq_len = 2160
n_features = 1
n_targets = 5

# generating training model
def generate_data(df, batch_size):
    num_curves = len(df['Index'].unique())
    num_batches_per_epoch = num_curves // batch_size

    while True:
        # mix the data
        shuffled_curves = np.random.permutation(df['Index'].unique())

        # foreach batch
        for i in range(num_batches_per_epoch):
            # random pick of batch_size curves
            batch_curves = shuffled_curves[i*batch_size:(i+1)*batch_size]

            # initializing the tensors for the batch
            X_batch = np.zeros((batch_size, seq_len, n_features))
            y_batch = np.zeros((batch_size, seq_len, n_targets))

            # filling the tensors
            for j, curve in enumerate(batch_curves):
                curve_data = df[df['Index'] == curve]
                X_batch[j,:,:] = curve_data[['watt']].values
                y_batch[j,:,:] = curve_data[['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']].values

            yield (X_batch, y_batch)

# creating the model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(seq_len, n_features)))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Reshape((1, 128)))
model.add(Conv1D(5, kernel_size=1, activation='sigmoid'))

# compiling the model
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# training the model
train_generator = generate_data(df_train, batch_size=batch_size)
model.fit(train_generator, steps_per_epoch=50, epochs=5)

# load the test data
df_test = pd.read_csv("InputTest.csv")
df_test = transform(df_test,'watt')

def generate_test_data(df):
    for _, curve_data in df.groupby('Index'):
        x = np.array(curve_data[['watt']].values).reshape(1, -1, 1)
        yield x

# predicting the data
y_pred = []
for x_test in generate_test_data(df_test):
    y_test_pred = model.predict(x_test)
    y_pred.append(y_test_pred[0,:,:])

# saving the data
result_df = pd.DataFrame(data=np.array(y_pred).reshape(-1, 5), columns=['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle'])
result_df = result_df.round(4)

result_df.to_csv('CNN_5epochs_50steps.csv', index=True,index_label="Index")
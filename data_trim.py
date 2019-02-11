from keras import layers 
from keras import models 
import numpy as np 
import pandas as pd
import keras
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

altitude = []
longitude = []

def read_data(fpath):
    train_df = pd.read_csv(fpath, header=0)
    # print(fpath + "finished reading. ")
    xl_length = len(train_df)
    # building + floor
    pos = []
    labels = []
    raw = []

    idx = np.arange(xl_length)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:,:]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:,0:520]).astype(float)
    
    for i in idx:
        row = rows[i]
        label = [int(float(row[520]) + 7465), int(float(row[521]) - 4864870)]
        # altitude.append(int(float(row[520]) + 7465))
        # longitude.append(int(float(row[521]) - 4864870))
        # label = int(float(row[520]) + 7465)
        signals = scale(sub_rows[i])
        pos.append(signals)
        labels.append(label)
        
    return np.array(pos), np.array(labels)

# read_data(fpath1)
# print(sum(altitude) / len(altitude), sum(longitude) / len(longitude))

def nn_model(model_name):
    if model_name == "DNN":
        model = keras.Sequential()
        model.add(keras.layers.Dense(520, activation='relu', input_shape=(520,)))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(2))

        model.compile(optimizer='rmsprop',
                      loss='mse',
                      metrics=['mae'])
        return model

def getPredictedBuildings():
    x, y = read_data(fpath1)
    x_test, y_test = read_data(fpath2)

    train_val_split = int(0.9 * len(x))  

    # train
    x_train = np.array(x[:train_val_split])
    y_train = np.array(y[:train_val_split])

    # validation
    x_val = np.array(x[train_val_split:])
    y_val = np.array(y[train_val_split:])

    model = nn_model("DNN")
    history = model.fit(x_train, y_train, epochs=11, batch_size=64, validation_data=(x_val, y_val))
    test_loss, test_acc = model.evaluate(x_test, y_test)

    mae_history = history.history['val_mean_absolute_error']
    plt.plot(range(1, len(mae_history) + 1), mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    # print('Test accuracy:', test_acc)
    pred = model.predict(x_test)

    for i in range(0, len(x_test)):
        alt = pred[i][0]
        longi = pred[i][1]

        difference =  np.sqrt(np.square(y_test[i][0] - alt) + np.square(y_test[i][1] - longi))
        print(i," : ", difference)

getPredictedBuildings()
from keras import layers 
from keras import models 
import numpy as np 
import pandas as pd
import keras

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

def one_hot_conversion(a):
    m = np.zeros(3)
    m[int(float(a))] = 1
    return m

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
        label = one_hot_conversion(int(float(row[523])))
        signals = (sub_rows[i] + 110) * 255 / 110
        pos.append(signals)
        labels.append(label)
        raw.append(row)
        
    return np.array(pos), np.array(labels), raw

def nn_model(model_name):
    if model_name == "DNN":
        model = keras.Sequential()
        model.add(keras.layers.Dense(520, activation='relu', input_shape=(520,)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(3, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

def getPredictedBuildings():
    x_train, y_train, train_raw = read_data(fpath1)
    x_test, y_test, test_raw = read_data(fpath2)

    model = nn_model("DNN")
    history = model.fit(x_train, y_train, epochs=2, batch_size=64)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    # print('Test accuracy:', test_acc)
    pred = model.predict(x_test)

    Buildings = {}
    Buildings_raw = {}
    count = 0
    for i in range(0, len(x_test)):
        if (np.argmax(pred[i]) == np.argmax(y_test[i])):
            count += 1
        key = np.argmax(pred[i])
        if key not in Buildings.keys():
            Buildings[key] = []
            Buildings_raw[key] = []
        Buildings[key].append(x_test[i])
        Buildings_raw[key].append(test_raw[i])
    print("Building accuracy: ", 100 * count / len(x_test), "%")
    return Buildings, Buildings_raw

# getPredictedBuildings()
from keras import layers 
from keras import models 
import numpy as np 
import pandas as pd
import keras
from keras.utils.np_utils import to_categorical
import BFP_Building
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"

def one_hot_conversion(a):
    m = np.zeros(5)
    m[int(float(a))] = 1
    return m

def read_data(fpath, method):
    train_df = pd.read_csv(fpath, header=0)
    data_len = len(train_df)

    buildings = {}
    building_labels = {}

    idx = np.arange(data_len)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:, :]).astype(float)
    sub_rows = scale(np.asarray(train_df.iloc[:, 0:520]).astype(float), axis=1)

    for i in idx:
        row = rows[i]
        building_number = int(float(row[523]))
        if building_number not in buildings.keys():
            buildings[building_number] = []
            building_labels[building_number] = []
        sub_row = sub_rows[i]
        buildings[building_number].append(sub_row)
        label = to_categorical(int(float(row[522])), num_classes=5)
        building_labels[building_number].append(label)
    return buildings, building_labels

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
        model.add(keras.layers.Dense(5, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

def getPredictedBFs():
    train_buildings, train_labels = read_data(fpath1, MODEL_NAME)
    test_buildings, test_raw = BFP_Building.getPredictedBuildings()
    BF_result = {}
    for key in train_buildings.keys():
        # 0-536-97%，1-307-84%， 2-268-94%
        print("Group ", key, " : ")
        x = train_buildings[key]
        y = train_labels[key]

        train_val_split = int(len(x))  # mask index array
        # train
        x_train = np.array(x[:train_val_split])
        y_train = np.array(y[:train_val_split])
        # # validation
        # x_val = np.array(x[train_val_split:])
        # y_val = np.array(y[train_val_split:])

        # test
        x_test = []
        y_test = []
        for i in range(0, len(test_buildings[key])):
            signals = scale(test_raw[key][i][:520])
            tmp = to_categorical(int(float(test_raw[key][i][522])), num_classes=5)
            x_test.append(signals)
            y_test.append(tmp)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)


        model = nn_model(MODEL_NAME)
        history = model.fit(x_train, y_train, epochs=2, batch_size=32)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print('Test accuracy:', test_acc)
        predictions = model.predict(x_test)

        correct_prediction = 0
        floor = [0,0,0,0,0]

        
        for i in range(0, len(x_test)):
            pred = np.argmax(predictions[i])
            bf_key = str(int(float(key))) + "-" + str(int(float(pred)))
            if bf_key not in BF_result.keys():
                BF_result[bf_key] = []
            BF_result[bf_key].append(test_raw[key][i])

            actl = np.argmax(y_test[i])
            if pred == actl:
                correct_prediction += 1
            else:
                floor[actl] += 1

        s = len(x_test)
        accuracy = 100 * correct_prediction / s

    print("For ", s, " test data:")
    print("floor accuracy: ", accuracy, "%")
    print("Total mistakes: ", sum(floor), " : ", floor)

    return BF_result
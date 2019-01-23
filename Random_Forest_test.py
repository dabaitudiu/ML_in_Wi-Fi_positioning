import xlrd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
import sklearn.feature_selection as fselection
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn.ensemble import RandomForestClassifier
from keras import models 
from keras import layers

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"
DIM = 409
SUB_DIM = 23
V_PARAM = 0.9

train_df = pd.read_csv(fpath1, header=0)
sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
sel = VarianceThreshold(threshold=(V_PARAM * (1 - V_PARAM)))
sel.fit(sub_rows)

def read_data(fpath, method):
    train_df = pd.read_csv(fpath, header=0)
    data_len = len(train_df)

    buildings = {}
    building_labels = {}

    idx = np.arange(data_len)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:, :]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
    sub_rows = sel.transform(sub_rows)
    print(sub_rows.shape)

    for i in idx:
        row = rows[i]
        building_number = int(float(row[523]))
        if building_number not in buildings.keys():
            buildings[building_number] = []
            building_labels[building_number] = []
        if method == "CNN":
            sub_x = (sub_rows[i].astype(float) + 110) / 110
            ax =[np.zeros(9)]
            sub_row = np.append(sub_x, ax)
            sub_row = sub_row.reshape(23, 23, 1)
        else:
            if method == "DNN":
                sub_row = (sub_rows[i].astype(float) + 110) / 110
            else:
                raise ValueError('Model unspecified.')
        buildings[building_number].append(sub_row)
        label = to_categorical(int(float(row[522])), num_classes=5)
        building_labels[building_number].append(label)
    return buildings, building_labels

train_buildings, train_labels = read_data(fpath1, MODEL_NAME)
test_buildings, test_labels = read_data(fpath2, MODEL_NAME)

for key in train_buildings.keys():
    print("Group ", key, " : ")

    x = train_buildings[key]
    y = train_labels[key]
    train_val_split = int(0.9 * len(x))  # mask index array
    # train
    x_train = np.array(x[:train_val_split])
    y_train = np.array(y[:train_val_split])
    # validation
    x_val = np.array(x[train_val_split:])
    y_val = np.array(y[train_val_split:])
    # test
    x_test = np.array(test_buildings[key])
    y_test = np.array(test_labels[key])

    # --------------------------------------------------------------------------------
    # Random Forest starts here.
    # --------------------------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=500)
    clf = clf.fit(x_train, y_train)
    a = clf.predict(x_test)
    count = 0
    for i in range(0,len(a)):
        if np.argmax(a[i]) == np.argmax(y_test[i]):
            count += 1
    print(count / len(a))

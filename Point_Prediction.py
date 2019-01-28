from keras import layers 
from keras import models 
import tensorflow as tf 
import numpy as np 
import pandas as pd
from keras.applications import VGG16 
from keras import models
from keras import layers
from keras import optimizers
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

def one_hot_conversion(a,b):
    m = np.zeros(3)
    n = np.zeros(5)
    m[int(float(a))] = 1
    n[int(float(b))] = 1
    tmp = np.append(m,n)
    return tmp

coordinates = {}

def read_data(fpath):
    train_df = pd.read_csv(fpath, header=0)
    # print(fpath + "finished reading. ")
    xl_length = len(train_df)
    # training data
    x = []
    # lables
    y_ = []
    # building + floor
    pos = {}
    count_label_number = {}

    idx = np.arange(xl_length)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:,:]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:,0:520]).astype(float)
    labels = {}

    # print("data finished processing - stage 1.")
    
    for i in idx:
        row = rows[i]
        key = str(int(float(row[523]))) + "-" +  str(int(float(row[522])))
        label = str(int(float(row[524]))) + "_" +  str(int(float(row[525])))
        signals = (sub_rows[i] + 110) * 255 / 110
        if key not in pos.keys():
            pos[key] = []
            labels[key] = []
            count_label_number[key] = []
        if label not in count_label_number[key]:
            count_label_number[key].append(label)
        label = count_label_number[key].index(label)
        pos[key].append(signals)
        labels[key].append(label)

    return pos,labels

train_keys, train_labels = read_data(fpath1)
# test_keys, test_labels = read_data(fpath2, MODEL_NAME)

results = {}
for key in train_keys.keys():
    x = train_keys[key]
    y = train_labels[key]
    train_val_split = int(0.9 * len(x))  # mask index array
    # train
    x_train = np.array(x[:train_val_split])
    y_train = np.array(y[:train_val_split])
    # validation
    x_test = np.array(x[train_val_split:])
    y_test = np.array(y[train_val_split:])
    # test
    # x_test = np.array(test_keys[key])
    # y_test = np.array(test_labels[key])

    selector= SelectKBest(score_func= f_classif, k=200)
    selector.fit(x_train, y_train)
    select_X = selector.transform(x_train)
    select_testX = selector.transform(x_test)

    clf = RandomForestClassifier(n_estimators=500)
    clf = clf.fit(select_X, y_train)
    # a = clf.predict_proba(select_testX)
    pred = clf.predict(select_testX)

    count = 0
    for i in range(len(select_testX)):
        if (pred[i] == y_test[i]):
            count += 1
    print(count / len(select_testX))

    if key not in results.keys():
        results[key] = []
    results[key].append(count / len(select_testX))

for key in results.keys():
    print(key," : ", results[key])

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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

def one_hot_conversion(a,b):
    m = np.zeros(3)
    n = np.zeros(5)
    m[int(float(a))] = 1
    n[int(float(b))] = 1
    tmp = np.append(m,n)
    return tmp


def read_data(fpath):
    train_df = pd.read_csv(fpath, header=0)
    # print(fpath + "finished reading. ")
    xl_length = len(train_df)
    # training data
    x = []
    # lables
    y_ = []

    pos = {}
    location = {}
    count_label_number = {}
    jw_position = {}

    idx = np.arange(xl_length)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:,:]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:,0:520]).astype(float)

    # print("data finished processing - stage 1.")
    
    for i in idx:
        row = rows[i]
        key = str(int(float(row[523]))) + "-" +  str(int(float(row[522])))
        label = str(int(float(row[524]))) + "-" +  str(int(float(row[525])))
        
        # print("original label is : ", label)
        signals = (sub_rows[i] + 110) * 255 / 110
        if key not in pos.keys():
            pos[key] = []
        pos[key].append(signals)
        # print("Key is ", key)
        if key not in location.keys():
            location[key] = {}
            count_label_number[key] = []
            jw_position[key] = {}
        # if label not in count_label_number[key]:
        #     count_label_number[key].append(label)
        # print("label is: ", label)
        # print("count_label_number[", key, "]: ", count_label_number[key])
        # label = count_label_number[key].index(label)
        # print("revised label is:", label)
        if label not in location[key].keys():
            location[key][label] = []
            jw_position[key][label] = []
        location[key][label].append(row)
        jw_position[key][label].append([row[520], row[521]])

    # for b_f in location.keys():
    #     signals_set = location[b_f]
    #     for point_label in signals_set.keys():
    #         print(b_f, "-", point_label, "-", len(signals_set[point_label]))
    return location, jw_position 

location, cord_all = read_data(fpath1)
final_data = []

for bf_key in location.keys():
    for point_key in location[bf_key].keys():
        coordinates = np.array(cord_all[bf_key][point_key])
        c_x = 0
        c_y = 0
        for i in range(len(coordinates)):
            c_x += coordinates[i][0]
            c_y += coordinates[i][1]
        if i == len(coordinates) - 1:
            c_x /= len(coordinates)
            c_y /= len(coordinates)

        print(bf_key, point_key, c_x, c_y)
        final_data.append([bf_key, point_key, c_x, c_y])
        # x_train = np.array(location[bf_key][point_key])
        # y_train = np.arange(len(location[bf_key][point_key]))

        # x_test = x_train
        # y_test = y_train 

        # selector= SelectKBest(score_func= f_classif, k=200)
        # selector.fit(x_train, y_train)
        # select_X = selector.transform(x_train)
        # select_testX = selector.transform(x_test)

        # clf = RandomForestClassifier(n_estimators=100)
        # clf = clf.fit(select_X, y_train)
        # a = clf.predict(select_testX)
        # count = 1
        # for i in range(0,len(a)):
        #     print(a[i])

final_data = pd.DataFrame(final_data)
print(final_data.dtypes)
# final_data.to_csv('final_data.csv',header=0, index=0)


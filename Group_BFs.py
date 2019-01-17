from keras import layers 
from keras import models 

import tensorflow as tf 
import numpy as np 
import pandas as pd
from keras.applications import VGG16 
from keras import models
from keras import layers
from keras import optimizers

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
    # dictionary: {building_floor:position}
    pos = {}
    location = {}
    count_label_number = {}

    idx = np.arange(xl_length)
    # np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:,:]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:,0:520]).astype(float)

    # print("data finished processing - stage 1.")
    
    for i in idx:
        row = rows[i]
        key = str(int(float(row[523]))) + "-" +  str(int(float(row[522])))
        label = str(int(float(row[524])))+ "-" +  str(int(float(row[525])))
        # print("original label is : ", label)
        signals = (sub_rows[i] + 110) * 255 / 110
        if key not in pos.keys():
            pos[key] = []
        pos[key].append(signals)
        # print("Key is ", key)
        if key not in location.keys():
            location[key] = {}
            count_label_number[key] = []
        if label not in count_label_number[key]:
            count_label_number[key].append(label)
        # print("label is: ", label)
        # print("count_label_number[", key, "]: ", count_label_number[key])
        label = count_label_number[key].index(label)
        # print("revised label is:", label)
        if label not in location[key].keys():
            location[key][label] = []
        location[key][label].append(row)
    
    for b_f in location.keys():
        signals_set = location[b_f]
        for point_label in signals_set.keys():
            print(b_f, "-", point_label, "-", len(signals_set[point_label]))

read_data(fpath1)
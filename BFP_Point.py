from keras import layers 
from keras import models 
import tensorflow as tf 
import numpy as np 
import pandas as pd
from keras import models
from keras import layers
from keras import optimizers
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import warnings
import BFP_Floor
from sklearn.preprocessing import scale

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

point_data = 'final_data.csv'
point_data_df = pd.read_csv(point_data, header=0)
point_data_df.columns = ['BF','Point','Latitude','Altitude']

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)


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
        label = str(int(float(row[524]))) + "-" +  str(int(float(row[525])))
        cord = [row[520], row[521]]
        signals = scale(sub_rows[i])
        if key not in pos.keys():
            pos[key] = []
            labels[key] = []
            count_label_number[key] = []
        if label not in count_label_number[key]:
            count_label_number[key].append(label)
        label = count_label_number[key].index(label)
        pos[key].append(signals)
        labels[key].append(label)
    return pos,labels,count_label_number

train_keys, train_labels, z0 = read_data(fpath1)
test_raw = BFP_Floor.getPredictedBFs()

# test_keys, test_labels = read_data(fpath2, MODEL_NAME)

results = {}
meters_results = {}
meters_error = []
for key in test_raw.keys():
    print("KEY: ", key, "------------------------------------------")
    x = train_keys[key]
    y = train_labels[key]
    train_val_split = int(len(x))  # mask index array
    # train
    x_train = np.array(x[:train_val_split])
    y_train = np.array(y[:train_val_split])

    # # validation
    # x_test = np.array(x[train_val_split:])
    # y_test = np.array(y[train_val_split:])

    # test
    x_test = []
    location_test = []
    for i in range(0, len(test_raw[key])):
        signals = scale(test_raw[key][i][:520])
        location = [test_raw[key][i][520], test_raw[key][i][521]]
        x_test.append(signals)
        location_test.append(location)
    x_test = np.array(x_test)
    print("x_test.shape = ", x_test.shape)

    selector= SelectKBest(score_func= f_classif, k=200)
    selector.fit(x_train, y_train)
    select_X = selector.transform(x_train)
    select_testX = selector.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(select_X, y_train)
    # a = clf.predict_proba(select_testX)
    pred = clf.predict(select_testX)

    count = 0
    for i in range(len(select_testX)):
        z = z0[key]
        # if (key == '2-4') or (key == '1-1'):
        #     continue
        condition1 = point_data_df['BF'] == key
        condition2 = point_data_df['Point'] == z[pred[i]]
        condition3 = condition1 & condition2 
        # print("Predicting ", key, z[pred[i]])
        pred_len = len(point_data_df[condition3].values)
        if (pred_len == 0):
            print("Error at ", key, z[pred[i]])
        pred_data_df = point_data_df[condition3].values[0]
        pred_altitude = pred_data_df[2]
        pred_longitude = pred_data_df[3]
        # print("predicted bf", key, "point", z[pred[i]], "at", pred_altitude, " , ", pred_longitude)

        # condition1 = point_data_df['BF'] == key
        # condition2 = point_data_df['Point'] == z[y_test[i]]
        # condition3 = condition1 & condition2 
        # test_data_df = point_data_df[condition3].values[0]
        # test_altitude = test_data_df[2]
        # test_longitude = test_data_df[3]

        test_altitude = location_test[i][0]
        test_longitude = location_test[i][1]

        # print("Actual point ", z[y_test[i]], "at", test_altitude, " , ", test_longitude)
        # print('-' * 50)
        
        difference = np.sqrt(np.square((pred_altitude - test_altitude)) + np.square((pred_longitude - test_longitude)))
        meters_error.append(difference)
#         # if (z[y_test[i]] != z[pred[i]]):
#         #     print(key, z[pred[i]], z[y_test[i]], pred_altitude, test_altitude, pred_longitude, test_longitude, difference)

#         new_key = key + "-" + z[y_test[i]]
#         if new_key not in meters_results.keys():
#             meters_results[new_key] = []
#         meters_results[new_key].append(difference)

#         if (pred[i] == y_test[i]):
#             count += 1
#     # print(key,count / len(select_testX))

#     if key not in results.keys():
#         results[key] = []
#     results[key].append(count / len(select_testX))

# for key in results.keys():
#     print(key," : ", results[key])

# total_sum = 0
# for item in meters_results.keys():
#     a = meters_results[item]
#     if len(a) != 0:
#         a_result = np.sum(a) / len(a)
#         print(item, " : ", a_result)
#         total_sum += a_result
# print(total_sum / len(x_train_0))

print("-----------------------------------------------------------------")

for i in range(0, len(meters_error)):
    print(i, " : ", meters_error[i])
print("Average error: ", sum(meters_error) / len(meters_error))
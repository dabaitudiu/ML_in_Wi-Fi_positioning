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
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from keras import models 
from keras import layers
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"
DIM = 409
SUB_DIM = 23
V_PARAM = 0.9

def read_data(fpath, method):
    train_df = pd.read_csv(fpath, header=0)
    data_len = len(train_df)

    buildings = {}
    building_labels = {}
    sparse_labels = {}

    idx = np.arange(data_len)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:, :]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
    min_max_scaler_0 = MinMaxScaler()
    sub_rows = min_max_scaler_0.fit_transform(sub_rows)
    # sub_rows = sel.transform(sub_rows)
    # print(sub_rows.shape)

    for i in idx:
        row = rows[i]
        building_number = int(float(row[523]))
        if building_number not in buildings.keys():
            buildings[building_number] = []
            building_labels[building_number] = []
            sparse_labels[building_number] = []
        if method == "CNN":
            sub_x = (sub_rows[i].astype(float) + 110) / 110
            ax =[np.zeros(9)]
            sub_row = np.append(sub_x, ax)
            sub_row = sub_row.reshape(23, 23, 1)
        else:
            if method == "DNN":
                # sub_row = (sub_rows[i].astype(float) + 110) / 110
                sub_row = sub_rows[i]
            else:
                raise ValueError('Model unspecified.')
        buildings[building_number].append(sub_row)
        sparse_label = to_categorical(int(float(row[522])), num_classes=5)
        label = int(float(row[522]))
        building_labels[building_number].append(label)
        sparse_labels[building_number].append(sparse_label)
    return buildings, building_labels, sparse_labels

train_buildings, train_labels, my_sparse_labels = read_data(fpath1, MODEL_NAME)
test_buildings, test_labels, my_test_sparse_labels = read_data(fpath2, MODEL_NAME)

key = 1
x = train_buildings[key]
y = train_labels[key]
y_2 = my_sparse_labels[key]
train_val_split = int(1 * len(x))  # mask index array

x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
y_2_train = np.array(y_2)

x_test = np.array(test_buildings[key])
y_test = np.array(test_labels[key])
y_2_test = np.array(my_test_sparse_labels[key])

selector= SelectKBest(score_func= f_classif, k=200)
warnings.simplefilter("ignore", category=FutureWarning)
selector.fit(x_train, y_train)
select_X = selector.transform(x_train)
select_testX = selector.transform(x_test)

# --------------------------------------------------------------------------------
# Random Forest starts here.
# --------------------------------------------------------------------------------

clf_forest = RandomForestClassifier(n_estimators=500, max_features=14)
clf_forest = clf_forest.fit(select_X, y_train)
a_forest = clf_forest.predict(select_testX)

count = 1
for i in range(0,len(a_forest)):
    if a_forest[i] == y_test[i]:
        count += 1
print("Forest: ", count / len(a_forest))

# --------------------------------------------------------------------------------
# KNN starts here.
# --------------------------------------------------------------------------------

clf_knn = KNeighborsClassifier(10)
clf_knn = clf_knn.fit(select_X, y_train)
a_knn = clf_knn.predict(select_testX)

count = 1
for i in range(0,len(a_knn)):
    if a_knn[i] == y_test[i]:
        count += 1
print("KNN: ", count / len(a_knn))

# --------------------------------------------------------------------------------
# Adaboost starts here.
# --------------------------------------------------------------------------------

clf_ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=10))
clf_ada = clf_ada.fit(select_X, y_train)
a_ada = clf_ada.predict(select_testX)

count = 1
for i in range(0,len(a_ada)):
    if a_ada[i] == y_test[i]:
        count += 1
print("Adaboost: ", count / len(a_ada))

# --------------------------------------------------------------------------------
# XGBoost starts here.
# --------------------------------------------------------------------------------

# params = {
#     'booster': 'gbtree',
#     'objective': 'multi:softmax',  # 多分类的问题
#     'num_class': 5,               # 类别数，与 multisoftmax 并用
#     'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#     'max_depth': 10,               # 构建树的深度，越大越容易过拟合
#     'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     'subsample': 0.7,              # 随机采样训练样本
#     'colsample_bytree': 0.7,       # 生成树时进行的列采样
#     'min_child_weight': 3,
#     'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
#     'eta': 0.01,                  # 如同学习率
#     'seed': 1000,
#     'n_estimators': 500,
#     'reshape': True,
#     'eval_metric': 'mlogloss'
# }

# dtrain = xgb.DMatrix(select_X, y_train)
# dtest = xgb.DMatrix(select_testX)
# num_rounds = 500
# model = xgb.train(params, dtrain, num_rounds)
# pred = model.predict(dtest)

# my_count = 0
# for i in range(len(select_testX)):
#     if (pred[i] == y_test[i]):
#         my_count += 1

# s = len(select_testX)
# accuracy = 100 * my_count / s

# print("For ", s, " test data:")
# print("floor accuracy: ", accuracy, "%")

# --------------------------------------------------------------------------------
# Neural Network starts here.
# --------------------------------------------------------------------------------
model = keras.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_shape=(520,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

history = model.fit(x_train,y_2_train, epochs=20, batch_size=64)
test_loss, test_acc = model.evaluate(x_test,y_2_test)
a_dnn = model.predict(x_test)
print('DNN:', test_acc)

count = 1
for i in range(len(a_forest)):
    if (np.argmax(a_dnn[i]) == 1):
        combined = [0,0,0,0,0]
        combined[a_forest[i]] += 1
        combined[a_knn[i]] += 1
        combined[a_ada[i]] += 1
        combined[np.argmax(a_dnn[i])] += 1
        result = np.argmax(combined)
    else:
        result = np.argmax(a_dnn[i])
    if result == y_test[i]:
        count += 1
print("Combined: ", count / len(a_ada))
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
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"
DIM = 409
SUB_DIM = 23
V_PARAM = 0.9

# train_df = pd.read_csv(fpath1, header=0)
# sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
# sel = VarianceThreshold(threshold=(V_PARAM * (1 - V_PARAM)))
# sel.fit(sub_rows)

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
    # sub_rows = sel.transform(sub_rows)
    print(sub_rows.shape)

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
                sub_row = (sub_rows[i].astype(float) + 110) / 110
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

key = 0
x = train_buildings[key]
y = train_labels[key]
y_2 = my_sparse_labels[key]
train_val_split = int(0.9 * len(x))  # mask index array
# train
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
y_2_train = np.array(y_2)

x_test = np.array(x[train_val_split:])
y_test = np.array(y[train_val_split:])

# x_test = np.array(test_buildings[key])
# y_test = np.array(test_labels[key])
# y_2_test = np.array(my_test_sparse_labels[key])

# for key in train_buildings.keys():
#     print("Group ", key, " : ")

#     x = train_buildings[key]
#     y = train_labels[key]
#     train_val_split = int(1 * len(x))  # mask index array
#     # train
#     x_train = np.array(x[:train_val_split])
#     y_train = np.array(y[:train_val_split])
#     # validation
#     # x_val = np.array(x[train_val_split:])
#     # y_val = np.array(y[train_val_split:])
#     # test
#     x_test = np.array(test_buildings[key])
#     y_test = np.array(test_labels[key])

    # --------------------------------------------------------------------------------
    # Random Forest starts here.
    # --------------------------------------------------------------------------------
    # clf = RandomForestClassifier(n_estimators=500)
    # clf = clf.fit(x_train, y_train)
    # a = clf.predict(x_test)
    # count = 0
    # for i in range(0,len(a)):
    #     if np.argmax(a[i]) == np.argmax(y_test[i]):
    #         count += 1
    # print(count / len(a))

selector= SelectKBest(score_func= f_classif, k=200)
warnings.simplefilter("ignore", category=FutureWarning)
# selector = RFE(estimator=RandomForestClassifier(n_estimators=10))
selector.fit(x_train, y_train)
select_X = selector.transform(x_train)
select_testX = selector.transform(x_test)

# print("N_features %s" % selector.n_features_) # 保留的特征数
# print("Ranking %s" % selector.ranking_) # 重要程度排名

# print("RFE finished.")
# Scores= selector.scores_
# # print(Scores)
# r = np.count_nonzero(~np.isnan(Scores))
# # g = selector.get_support(True)
# print("non-zero: ", r)

# --------------------------------------------------------------------------------
# Random Forest starts here.
# --------------------------------------------------------------------------------

clf = RandomForestClassifier(n_estimators=500)
# clf = KNeighborsClassifier(10)
clf = clf.fit(select_X, y_train)
a = clf.predict(select_testX)
count = 1
for i in range(0,len(a)):
    if a[i] == y_test[i]:
        count += 1
print(count / len(a))

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
#     'eta': 0.007,                  # 如同学习率
#     'seed': 1000,
#     'n_estimators': 1000,
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
# model = keras.Sequential()
# model.add(keras.layers.Dense(128, activation='relu', input_shape=(200,)))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(16, activation='relu'))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(5, activation='softmax'))

# model.compile(optimizer='rmsprop',
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

# history = model.fit(select_X, y_2_train, epochs=20, batch_size=64)
# test_loss, test_acc = model.evaluate(select_testX, y_2_test)
# print('Test accuracy:', test_acc)
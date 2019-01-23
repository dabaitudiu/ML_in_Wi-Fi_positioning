from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import sklearn.feature_selection as fselection
from keras import models 
from keras import layers
from keras import optimizers
from xgboost import plot_importance


params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 5,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 10,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                  # 如同学习率
    'seed': 1000,
    'n_estimators': 1000,
}


fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"

train_df = pd.read_csv(fpath1, header=0)
sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
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
        if method == "DNN":
            sub_row = (sub_rows[i].astype(float) + 110) / 110
        if method == "VGG16":
            sub_x = (sub_rows[i].astype(float) + 110) / 110
            sub_row = np.append(sub_x, sub_x)[:1024]
            sub_row = sub_row.reshape(32, 32, 1)
            tmp = np.append(sub_row, sub_row,axis=2)
            sub_row = np.append(tmp, sub_row,axis=2)
        if method != "CNN" and method != "DNN" and method != "VGG16":
            raise ValueError('Model unspecified.')
        buildings[building_number].append(sub_row)
        # label = to_categorical(int(float(row[522])), num_classes=5)
        label = int(float(row[522]))
        building_labels[building_number].append(label)
    return buildings, building_labels


train_buildings, train_labels = read_data(fpath1, MODEL_NAME)
test_buildings, test_labels = read_data(fpath2, MODEL_NAME)

key = 2
x = train_buildings[key]
y = train_labels[key]
train_val_split = int(1 * len(x))  # mask index array
# train
x_train = np.array(x[:train_val_split])
print(x_train.shape)
y_train = np.array(y[:train_val_split])
print(y_train.shape)
# validation
# x_val = np.array(x[train_val_split:])
# y_val = np.array(y[train_val_split:])
# test
x_test = np.array(test_buildings[key])
y_test = np.array(test_labels[key])

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
num_rounds = 500
model = xgb.train(params, dtrain, num_rounds)
pred = model.predict(dtest)

# gbm = xgb.XGBClassifier(silent=0, max_depth=8, n_estimators=100, learning_rate=0.1)
# gbm.fit(x_train, y_train)
# pred = gbm.predict(x_train)

count_c = 0

print(pred)
count = 0
for i in range(len(x_test)):
    if (pred[i] == y_test[i]):
        count += 1
    else:
        if (y_test[i] == 1):
            count_c += 1
            print(y_test[i],"->",pred[i])

print(count / len(x_test))
plot_importance(model)
plt.show()
print("count_c = ", count_c)
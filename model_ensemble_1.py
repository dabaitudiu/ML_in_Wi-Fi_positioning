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
    'objective': 'multi:softprob',  # 多分类的问题
    'num_class': 5,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 10,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                  # 如同学习率
    'seed': 1000,
    'n_estimators': 1500,
}


fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"

train_df = pd.read_csv(fpath1, header=0)
sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
sel = VarianceThreshold(threshold=0)
sel.fit(sub_rows)

def read_data(fpath, method):
    train_df = pd.read_csv(fpath, header=0)
    data_len = len(train_df)

    buildings = {}
    building_labels = {}
    integer_labels = {}

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
            integer_labels[building_number] = []
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
        label = to_categorical(int(float(row[522])), num_classes=5)
        int_label = int(float(row[522]))
        building_labels[building_number].append(label)
        integer_labels[building_number].append(int_label)
    return buildings, building_labels, integer_labels

def nn_model(model_name):
    if model_name == "DNN":
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='relu', input_shape=(465,)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(128, activation='relu'))
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
    if model_name == "CNN":
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(23, 23, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(5, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    if model_name == "VGG16":
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(5, activation='sigmoid'))

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
    if model_name == "InceptionV3":
        conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(32,32,3))
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(5, activation='sigmoid'))

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
    return model

train_buildings, train_labels, train_labels_int = read_data(fpath1, MODEL_NAME)
test_buildings, test_labels, test_labels_int = read_data(fpath2, MODEL_NAME)

# ------------------------------------------------------------------------------
# DNN Section
#-------------------------------------------------------------------------------
key = 0
x = train_buildings[key]
y = train_labels[key]
train_val_split = int(1 * len(x))  # mask index array
# train
x_train = np.array(x[:train_val_split])
print(x_train.shape)
y_train = np.array(y[:train_val_split])
print(y_train.shape)
# test
x_test = np.array(test_buildings[key])
y_test = np.array(test_labels[key])

model = nn_model(MODEL_NAME)
history = model.fit(x_train, y_train, epochs=20, batch_size=64)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(x_test)

# ------------------------------------------------------------------------------
# XGBoost Section
#-------------------------------------------------------------------------------
print('-' * 50)
print("XGBoost started.")

x = train_buildings[key]
y = train_labels_int[key]
train_val_split = int(1 * len(x))  # mask index array
# train
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
# test
x_test = np.array(test_buildings[key])
y_test = np.array(test_labels_int[key])

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
num_rounds = 500
model = xgb.train(params, dtrain, num_rounds)
pred = model.predict(dtest)

print("XGBoost finished.")

combined_result = []
alpha = 0.5
beta = 0.5

nn_wrong = 0
xgb_wrong = 0
both_wrong = 0
nn_correct = 0
xgb_correct = 0

for i in range(len(x_test)):    
    result1 = np.argmax(predictions[i])
    result2 = np.argmax(pred[i])
    actual_result = y_test[i]
    if (result1 == actual_result):
        nn_correct += 1
    if (result2 == actual_result):
        xgb_correct += 1
    if (result1 != result2):
        print("Case: ", i)
        print("NN Outcome is: ", predictions[i], " - ", result1)
        print("XGB Outcome is: ", pred[i],  " - ", result2)
        tmp = alpha * pred[i] + beta * predictions[i]
        print("Combine Outcome is: ", tmp)
        f_result = np.argmax(tmp)
        print("Final Predicted Index is: ", f_result)
        print("actual Index is: ", actual_result)
        if (result1 == actual_result):
            xgb_wrong += 1
        else:
            if (result2 == actual_result):
                nn_wrong += 1
            else:
                both_wrong += 1
                print("both wrong happened at case: ", i)
    else:
        tmp = alpha * pred[i] + beta * predictions[i]
    combined_result.append(tmp)

# gbm = xgb.XGBClassifier(silent=0, max_depth=8, n_estimators=100, learning_rate=0.1)
# gbm.fit(x_train, y_train)
# pred = gbm.predict(x_train)

correct_prediction = 0
floor = [0,0,0,0,0]

for i in range(len(x_test)):
    final_pred = np.argmax(combined_result[i])
    actl = y_test[i]
    if final_pred == actl:
        correct_prediction += 1
    else:
        floor[actl] += 1

s = len(x_test)
accuracy = 100 * correct_prediction / s

print('-'*50)
print("For ", s, " test data:")
print("floor accuracy: ", accuracy, "%")
print("Total mistakes: ", sum(floor), " : ", floor)

print('-'*50)
print("NN wrong: ", nn_wrong)
print("XGB wrong: ", xgb_wrong)
print("Both wrong: ", both_wrong)

print('-'*50)
print("NN accuracy: ", nn_correct)
print("XGB accuracy: ", xgb_correct)
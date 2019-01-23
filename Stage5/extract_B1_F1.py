import xlrd
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
from keras.applications import VGG16 
from keras.applications import InceptionV3
import scipy.stats
import os 


fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"
INPUT_DIM = 520

save_dir = 'MNIST_data/extract_B1F1/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# train_df = pd.read_csv(fpath1, header=0)
# sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
# sel = VarianceThreshold(threshold=0)
# sel.fit(sub_rows)

sub_floors = [[],[],[],[]]

def read_data(fpath, method):
    train_df = pd.read_csv(fpath, header=0)
    data_len = len(train_df)

    buildings = {}
    building_labels = {}

    idx = np.arange(data_len)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:, :]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
    # sub_rows = sel.transform(sub_rows)
    
    ind_k = 0

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
            if building_number == 1 and fpath == "validationData2.csv":
                tmp_label = int(float(row[522]))
                sub_floors[tmp_label].append(sub_row)
                if tmp_label == 1:
                    ind_k += 1
                    ax = [np.zeros(9)]
                    concax = np.append(sub_row,ax)
                    concax = concax.reshape(23,23)
                    filename = save_dir + '%d.jpg'%ind_k
                    scipy.misc.toimage(concax, cmin=0.0, cmax=1.0).save(filename)
        if method == "VGG16" or method =="S_CNN":
            sub_x = (sub_rows[i].astype(float) + 110) / 110
            sub_row = np.append(sub_x, sub_x)[:1024]
            sub_row = sub_row.reshape(32, 32, 1)
            tmp = np.append(sub_row, sub_row,axis=2)
            sub_row = np.append(tmp, sub_row,axis=2)
        if method != "CNN" and method != "DNN" and method != "VGG16" and method != "S_CNN":
            raise ValueError('Model unspecified.')
        buildings[building_number].append(sub_row)
        label = to_categorical(int(float(row[522])), num_classes=5)
        building_labels[building_number].append(label)
    return buildings, building_labels


def nn_model(model_name):
    if model_name == "DNN":
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='relu', input_shape=(INPUT_DIM,)))
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
        model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(23, 23, 1)))
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

    if model_name == "S_CNN":
        model = models.Sequential()
        model.add(layers.SeparableConv2D(32,3,activation='relu',input_shape=(32,32,3)))
        model.add(layers.SeparableConv2D(64,3,activation='relu'))
        model.add(layers.MaxPooling2D(2))
        
        model.add(layers.SeparableConv2D(64,3,activation='relu'))
        model.add(layers.SeparableConv2D(128,3,activation='relu'))
        model.add(layers.MaxPooling2D(2))

        model.add(layers.SeparableConv2D(64,3,activation='relu'))
        model.add(layers.SeparableConv2D(128,3,activation='relu'))
        model.add(layers.GlobalAveragePooling2D())

        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model


# train_buildings, train_labels = read_data(fpath1, MODEL_NAME)
# test_buildings, test_labels = read_data(fpath2, MODEL_NAME)

# # key = 1
# # x = train_buildings[key]
# # y = train_labels[key]
# # train_val_split = int(0.9 * len(x))  # mask index array
# # # train
# # x_train = np.array(x[:train_val_split])
# # y_train = np.array(y[:train_val_split])
# # # validation
# # x_val = np.array(x[train_val_split:])
# # y_val = np.array(y[train_val_split:])
# # # test
# # x_test = np.array(test_buildings[key])
# # y_test = np.array(test_labels[key])

# model = nn_model(MODEL_NAME)
# history = model.fit(x_train, y_train, epochs=20, batch_size=512)
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('Test accuracy:', test_acc)
# predictions = model.predict(x_test)

# correct_prediction = 0
# floor = [0,0,0,0,0]
# count_0 = 0
# count_2 = 0

# for i in range(1, len(x_test)):
#     pred = np.argmax(predictions[-i])
#     actl = np.argmax(y_test[-i])
#     if pred == actl:
#         correct_prediction += 1
#     else:
#         floor[actl] += 1
#         if (actl == 1):
#             if (pred == 0):
#                 count_0 += 1
#             if (pred == 2):
#                 count_2 += 1

# s = len(x_test)
# accuracy = 100 * correct_prediction / s

# print("For ", s, " test data:")
# print("floor accuracy: ", accuracy, "%")
# print("Total mistakes: ", sum(floor), " : ", floor)
# print("1 count as 0: ", count_0);
# print("1 count as 2: ", count_2)


# for item in sub_floors:
#     item = np.array(item)
#     print(item.shape)



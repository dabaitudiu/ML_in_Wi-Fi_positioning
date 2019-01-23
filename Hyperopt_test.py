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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras import losses
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.models import Sequential
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras.layers import Input
import sys





fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

MODEL_NAME = "DNN"

# train_df = pd.read_csv(fpath1, header=0)
# sub_rows = np.asarray(train_df.iloc[:, 0:520]).astype(float)
# sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
# sel.fit(sub_rows)

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
        building_labels[building_number].append(label)
    return buildings, building_labels

def nn_model(model_name):
    if model_name == "DNN":
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='relu', input_shape=(520,)))
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

space = {
        'units1': hp.choice('units1', [256, 512]),
        'units2': hp.choice('units2', [64, 128, 256]),
        'units3': hp.choice('units3', [32, 64, 128]),
        'units4': hp.choice('units4', [16, 32, 64]),
        'dropout':hp.choice('dropout', [0.2, 0.3, 0.4, 0.5]),

        'lr': hp.choice('lr',[0.01, 0.001, 0.0001]),
        'activation': hp.choice('activation',['relu',
                                                'sigmoid',
                                                'tanh',
                                                'linear']),
        'loss': hp.choice('loss', [losses.logcosh,
                                    losses.mse,
                                    losses.mae,
                                    losses.mape])
        }

train_buildings, train_labels = read_data(fpath1, MODEL_NAME)
test_buildings, test_labels = read_data(fpath2, MODEL_NAME)

key = 2
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

# def experiment(params):
#     X_train = x_train
#     X_test = x_test 
#     Y_train = y_train 
#     Y_test = y_test
#     print ('Trying', params)
    
#     main_input = Input(shape=(len(X_train[0]), ), name='main_input')
#     x = Dense(params['units1'], activation=params['activation'])(main_input)
#     x = Dropout(params['dropout'])(x)
#     x = Dense(params['units2'], activation=params['activation'])(x)
#     x = Dropout(params['dropout'])(x)
#     x = Dense(params['units3'], activation=params['activation'])(x)
#     x = Dropout(params['dropout'])(x)

#     output = Dense(5, activation = "softmax", name = "out")(x)
#     final_model = Model(inputs=[main_input], outputs=[output])
#     opt = Adam(lr=params['lr'])

#     final_model.compile(optimizer=opt,  loss=params['loss'])

#     history = final_model.fit(X_train, Y_train, 
#             epochs = 10, 
#             batch_size = 256, 
#             verbose=1, 
#             validation_data=(x_val, y_val),
#             shuffle=True)

#     pred = final_model.predict(X_test)


#     predicted = pred
#     original = Y_test

#     mse = np.mean(np.square(predicted - original))    

#     if np.isnan(mse):
#         print ('NaN happened')
#         print ('-' * 10)
#         return {'loss': 999999, 'status': STATUS_OK}

#     print (mse)
#     print ('-' * 10)

#     sys.stdout.flush() 
#     return {'loss': mse, 'status': STATUS_OK}

# trials = Trials()
# best = fmin(experiment, space, algo=tpe.suggest, max_evals=50, trials=trials)
# print ('best: ')
# print (best)


main_input = Input(shape=(len(x_train[0]), ), name='main_input')
x = Dense(512, activation='relu')(main_input)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(8, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(5, activation = "softmax", name = "out")(x)
final_model = Model(inputs=[main_input], outputs=[output])
opt = Adam(lr=0.001)

final_model.compile(optimizer=opt,  loss=losses.mape)

history = final_model.fit(x_train, y_train, 
        epochs = 10, 
        batch_size = 256, 
        verbose=1, 
        validation_data=(x_val, y_val),
        shuffle=True)

predictions = final_model.predict(x_test)

correct_prediction = 0
floor = [0,0,0,0,0]

for i in range(1, len(x_test)):
    pred = np.argmax(predictions[-i])
    actl = np.argmax(y_test[-i])
    if pred == actl:
        correct_prediction += 1
    else:
        floor[actl] += 1

s = len(x_test)
accuracy = 100 * correct_prediction / s

print("For ", s, " test data:")
print("floor accuracy: ", accuracy, "%")

    

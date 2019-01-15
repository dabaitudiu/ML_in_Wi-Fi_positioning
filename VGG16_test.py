import numpy as np 
import scipy.misc 
import os 
import xlrd 
import matplotlib.pyplot as plt 
from keras.applications import VGG16 
from tensorflow import keras 
import rssi_data as data
import read_test_data as test_data 
import tensorflow as tf 
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Model


save_dir = 'MNIST_data/images/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

def one_hot_conversion(a,b,c):
    m = np.zeros(3)
    n = np.zeros(5)
    q = np.zeros(110)
    m[int(float(a))] = 1
    n[int(float(b))] = 1
    q[int(float(c))] = 1
    tmp = np.append(m,n)
    tmp = np.append(tmp,q)
    return tmp

fpath1 = "MNIST_data/test/datav2.xlsx"
fpath2 = "MNIST_data/test/validationData2.xlsx"

def read_data(fpath):
    xl_book = xlrd.open_workbook(fpath)
    xl_table = xl_book.sheets()[0] 
    xl_length = xl_table.nrows
    # training data
    x = []
    # lables
    y_ = []
    # dictionary: {building_floor:position}
    pos = {}

    idx = np.arange(1, xl_length)
    np.random.shuffle(idx)

    for i in idx:
        row = np.array(xl_table.row_values(i))
        # modification on row[524] to make it range from 0 to 110 rather random number.
        myKey = str(row[523]) + "-" + str(row[522])
        if myKey not in pos.keys():
            pos[myKey] = []
        if row[524] not in pos[myKey]:
            pos[myKey].append(row[524])
        row[524] = pos[myKey].index(row[524])

        # make inputs range from 0 ~ 255 
        signals = (np.array(xl_table.row_values(i)[:520]).astype(np.float64) + 110) * 255 / 110
        signals = np.append(signals,signals)[:1024]
        signals = signals.reshape(32,32,1)
        tmp = np.append(signals,signals,axis=2)
        signals = np.append(tmp,signals,axis=2)

        label = one_hot_conversion(row[523],row[522],row[524])

        # append training data and lables
        x.append(signals)
        y_.append(label)


    x = np.array(x)
    y_ = np.array(y_)

    return x, y_

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

conv_base.summary()

x,y = read_data(fpath1)
train_val_split = int(0.9 * len(x))  # mask index array
# train
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
# validation
x_val = np.array(x[train_val_split:])
y_val = np.array(y[train_val_split:])
# test

x_test, y_test = read_data(fpath2)
x_test = np.array(x_test)
y_test = np.array(y_test)

# fc0 = layers.Flatten(name='flatten')(conv_base.outputs)
# fc = layers.Dense(512,activation='relu')(fc0)
# drop = layers.Dropout(0.5)(fc)
# fc = layers.Dense(256,activation='relu')(drop)
# drop = layers.Dropout(0.5)(fc)
# fc2 = layers.Dense(118,activation='sigmoid')(drop)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(118,activation='sigmoid'))

model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

print(x.shape)
print(y.shape)
print(x_val.shape)
print(y_val.shape)

history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=20,
                    validation_data=(x_val,y_val))

test_loss, test_acc = model.evaluate(x_test,y_test)

print('Test accuracy:', test_acc)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

predictions = model.predict(x_test)

b = 0
b_f = 0
b_f_p = 0

for i in range(1,len(x_test)):
    sub1 = np.argmax(predictions[-i][:3])
    sub2 = np.argmax(predictions[-i][3:8])
    sub3 = np.argmax(predictions[-i][8:118])

    sub4 = np.argmax(y_test[-i][:3])
    sub5 = np.argmax(y_test[-i][3:8])
    sub6 = np.argmax(y_test[-i][8:118])

    if (sub1 == sub4) and (sub2 == sub5) and (sub3 == sub6):
        b_f_p += 1
        b_f += 1
        b += 1
    else:
        if (sub1 == sub4) and (sub2 == sub5):
            b_f += 1
            b += 1
        else:
            if (sub1 == sub4):
                b += 1

s = len(x_test)
a1 = 100 * b / s
a2 = 100 * b_f / s
a3 = 100 * b_f_p / s

print("For ",s," test data:")
print("building accuracy: ", a1,"%")
print("building + floor prediction accuracy: ", a2,"%")
print("building + floor + place accuracy: ", a3,"%")




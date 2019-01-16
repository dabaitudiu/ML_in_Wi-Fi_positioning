from keras import layers 
from keras import models 
import rssi_data as train_data 
import read_test_data as test_data 
import tensorflow as tf 
import numpy as np 
import pandas as pd


def one_hot_conversion(a,b):
    m = np.zeros(3)
    n = np.zeros(5)
    m[int(float(a))] = 1
    n[int(float(b))] = 1
    tmp = np.append(m,n)
    return tmp

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

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

    idx = np.arange(xl_length)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:,:]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:,0:520]).astype(float)

    # print("data finished processing - stage 1.")

    for i in idx:
        row = rows[i]
        label = one_hot_conversion(row[523],row[522])
        y_.append(label)

        row = (sub_rows[i] + 110) * 255 / 110
        ax = [np.zeros(9)]
        sub_x = np.append(row, ax)
        # print("sub_x medium shape: ", sub_x.shape)
        sub_x = sub_x.reshape(23,23,1)
        # print("sub_x modified shape: ", sub_x.shape)
        x.append(sub_x)

    # print("data finished processing - stage 2.")
    x = np.array(x)
    y_ = np.array(y_)

    return x, y_

x, y = read_data(fpath1)
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


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(23,23,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(8, activation='sigmoid'))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=20,batch_size=64)

test_loss, test_acc = model.evaluate(x_test,y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(x_test)

b = 0
b_f = 0

for i in range(1,len(x_test)):
    sub1 = np.argmax(predictions[-i][:3])
    sub2 = np.argmax(predictions[-i][3:8])

    sub4 = np.argmax(y_test[-i][:3])
    sub5 = np.argmax(y_test[-i][3:8])

    if (sub1 == sub4) and (sub2 == sub5):
        b_f += 1
        b += 1
    else:
        if (sub1 == sub4):
            b += 1

s = len(x_test)
a1 = 100 * b / s
a2 = 100 * b_f / s

print("For ",s," test data:")
print("building accuracy: ", a1,"%")
print("building + floor prediction accuracy: ", a2,"%")



import xlrd 
import numpy as np 
from tensorflow import keras 
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.utils.np_utils import to_categorical

fpath = "MNIST_data/test/train_3000.xlsx"

filepath = fpath
xl_book = xlrd.open_workbook(filepath)
xl_table = xl_book.sheets()[0]
xl_length = xl_table.nrows

# {building_floor:[[inputs[0],inputs[1],...],[labels[0],labels[1],labels[2]...]]}
loc_sets = {}
# rearrange row[524]
pos = {}

idx = np.arange(1, xl_length)
np.random.shuffle(idx)

for i in idx:
    row = np.array(xl_table.row_values(i))
    building_number = int(float(row[523]))
    floor_number = int(float(row[522]))
    
    # generate key in the loc_sets dictionary.
    loc_sets_label = str(building_number) + "_" + str(floor_number)
    if loc_sets_label not in loc_sets.keys():
        loc_sets[loc_sets_label] = [[],[]]

    # move inputs and labels into a single array, 
    # retrieve inputs using loc_sets_contents[0], labels using loc_sets[1]
    loc_sets_contents = []
    # regularization, making inputs ranging from 0 - 1
    regularized_row = (row[:520].astype(np.float64)+110)/110
    # append inputs
    loc_sets_contents.append(regularized_row)
    # rearrange row[524] to 0~110 and generate one-hot label using keras function
    myKey = loc_sets_label
    if myKey not in pos.keys():
        pos[myKey] = []
    if row[524] not in pos[myKey]:
        pos[myKey].append(row[524])
    row[524] = pos[myKey].index(row[524])
    label = to_categorical(int(float(row[524])),num_classes=110)
    # append label
    loc_sets_contents.append(label)
    loc_sets[loc_sets_label][0].append(regularized_row)
    loc_sets[loc_sets_label][1].append(label)

# convert all lists to numpy arrays

for item in loc_sets:
    loc_sets[item][0] = np.array(loc_sets[item][0])
    loc_sets[item][1] = np.array(loc_sets[item][1])

x = loc_sets['2_3'][0]
y = loc_sets['2_3'][1]
train_val_split = int(0.8 * len(x))  # mask index array
val_test_split = int(0.9 * len(x))
# train
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
# validation
x_val = np.array(x[train_val_split:val_test_split])
y_val = np.array(y[train_val_split:val_test_split])
# test
x_test = np.array(x[val_test_split:])
y_test = np.array(y[val_test_split:])


print("data shapes:")
print(x.shape)
print(y.shape)
print("train shapes:")
print(x_train.shape)
print(y_train.shape)

model = keras.Sequential()
model.add(keras.layers.Dense(256, activation='relu', input_shape=(520,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(110, activation='softmax'))



model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20,batch_size=512,validation_data=(x_val,y_val))

test_loss, test_acc = model.evaluate(x_test,y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(x_test)

correct_prediction = 0

for i in range(1,len(x_test)):
    pred = np.argmax(predictions[-i])
    actl = np.argmax(y_test[-i])
    if pred == actl:
        correct_prediction += 1
   

s = len(x_test)
accuracy = 100 * correct_prediction / s 

print("For ",s," test data:")
print("floor accuracy: ", accuracy,"%")

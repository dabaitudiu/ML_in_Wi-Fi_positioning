import xlrd 
import numpy as np 
from tensorflow import keras 
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.utils.np_utils import to_categorical

fpath = "MNIST_data/test/datav2.xlsx"

filepath = fpath
xl_book = xlrd.open_workbook(filepath)
xl_table = xl_book.sheets()[0]
xl_length = xl_table.nrows

# training data
x = []
# lables
y_ = []
# dictionary: {building_floor:position}
pos = {}

building_signals = [[], [], []]
building_labels = [[], [], []]

idx = np.arange(1, xl_length)
np.random.shuffle(idx)

for i in idx:
    row = np.array(xl_table.row_values(i))
    building_number = int(float(row[523]))
    building_signals[building_number].append((row[:520].astype(np.float64)+110)/110)    
    label = to_categorical(int(float(row[522])),num_classes=5)
    building_labels[building_number].append(label)

for i in range(3):
    building_signals[i] = np.array(building_signals[i])
    building_labels[i] = np.array(building_labels[i])
    print(building_signals[i].shape)
    print(building_labels[i].shape)


# train building 0:-----------------------------------------------------------------------------------------------------------------------------------------
print("Building 0 Training:")
x = building_signals[1]
y = building_labels[1]

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

history = model.fit(x_train, y_train, epochs=11,batch_size=512,validation_data=(x_val,y_val))

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

with open('MNIST_data/test/record.txt', 'a+') as f:
        str1 = "With Shuffled: \n" + "Floor prediction accuracy at Building 0: " + str(accuracy) + "%\n"
        f.write(str1)

print("For ",s," test data:")
print("floor accuracy: ", accuracy,"%")

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

# train building 1:-----------------------------------------------------------------------------------------------------------------------------------------
print("Building 1 Training:")
x = building_signals[1]
y = building_labels[1]

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

history = model.fit(x_train, y_train, epochs=11,batch_size=512,validation_data=(x_val,y_val))

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

with open('MNIST_data/test/record.txt', 'a+') as f:
        str1 = "Floor prediction accuracy at Building 1: " + str(accuracy) + "%\n"
        f.write(str1)

print("For ",s," test data:")
print("floor accuracy: ", accuracy,"%")

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

# train building 2: --------------------------------------------------------------------------------------------------------------------------------------------------
print("Building 2 Training:")
x = building_signals[2]
y = building_labels[2]

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

history = model.fit(x_train, y_train, epochs=16,batch_size=512,validation_data=(x_val,y_val))

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

with open('MNIST_data/test/record.txt', 'a+') as f:
        str1 = "Floor prediction accuracy at Building 2: " + str(accuracy) + "%\n"
        f.write(str1)

print("For ",s," test data:")
print("floor accuracy: ", accuracy,"%")

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

import xlrd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras import models 
from keras import layers 

fpath = "MNIST_data/test/datav2.xlsx"
fpath2 = "MNIST_data/test/validationData2.xlsx"

filepath = fpath
xl_book = xlrd.open_workbook(filepath)
xl_table = xl_book.sheets()[0]
xl_length = xl_table.nrows

filepath2 = fpath2
xl_book2 = xlrd.open_workbook(filepath2)
xl_table2 = xl_book2.sheets()[0]
xl_length2 = xl_table2.nrows

# training data
x = []
# lables
y_ = []


# test data
x_test = []
# lables
y_test = []


building_signals = [[], [], []]
building_labels = [[], [], []]

building_signals2 = [[], [], []]
building_labels2 = [[], [], []]

for i in range(1, xl_length):
    row = np.array(xl_table.row_values(i))
    building_number = int(float(row[523]))
    sub_x = (row[:520].astype(np.float64)+110)/110
    ax = [np.zeros(9)]
    sub_x = np.append(sub_x, ax)
    sub_x = sub_x.reshape(23,23,1)
    building_signals[building_number].append(sub_x)
    label = to_categorical(int(float(row[522])), num_classes=5)
    building_labels[building_number].append(label)

for i in range(3):
    building_signals[i] = np.array(building_signals[i])
    building_labels[i] = np.array(building_labels[i])
    print(building_signals[i].shape)
    print(building_labels[i].shape)

for i in range(1, xl_length2):
    row2 = np.array(xl_table2.row_values(i))
    building_number2 = int(float(row2[523]))
    sub_x = (row2[:520].astype(np.float64)+110)/110
    ax = [np.zeros(9)]
    sub_x = np.append(sub_x, ax)
    sub_x = sub_x.reshape(23,23,1)
    building_signals2[building_number2].append(sub_x)
    label2 = to_categorical(int(float(row2[522])), num_classes=5)
    building_labels2[building_number2].append(label2)

for i in range(3):
    building_signals2[i] = np.array(building_signals2[i])
    building_labels2[i] = np.array(building_labels2[i])
    print(building_signals2[i].shape)
    print(building_labels2[i].shape)

# train buildings: --------------------------------------------------------------------------------------------------------------------------------------------------
for i in range(3):
    print("Building ",i, " Starts Training:")
    x = building_signals[i]
    y = building_labels[i]

    x_test = np.array(building_signals2[i])
    y_test = np.array(building_labels2[i])

    train_val_split = int(0.9 * len(x))  # mask index array

    # train
    x_train = np.array(x[:train_val_split])
    y_train = np.array(y[:train_val_split])
    # validation
    x_val = np.array(x[train_val_split:])
    y_val = np.array(y[train_val_split:])

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(23,23,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=30, batch_size=512, validation_data=(x_val, y_val))

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Test accuracy:', test_acc)

    predictions = model.predict(x_test)

    correct_prediction = 0

    for i in range(1, len(x_test)):
        pred = np.argmax(predictions[-i])
        actl = np.argmax(y_test[-i])
        if pred == actl:
            correct_prediction += 1

    s = len(x_test)
    accuracy = 100 * correct_prediction / s

    with open('MNIST_data/test/record.txt', 'a+') as f:
        str1 = "Floor prediction accuracy at Building 0: " + str(accuracy) + "%\n"
        f.write(str1)

    print("For ", s, " test data:")
    print("floor accuracy: ", accuracy, "%")

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    # Validation loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()

    # Validation accuracy
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    plt.clf()

    # # Test loss
    # plt.plot(epochs, test_acc, 'b', label='Test loss')
    # plt.title('Test accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.show()
    # plt.clf()



from keras import layers
from keras import Input
from keras.models import Model
import numpy as np
import pandas as pd 


def one_hot_conversion(ind, num):
    ind = int(float(ind))
    m = np.zeros(num)
    m[ind] = 1
    return m

fpath1 = "trainingData2.csv"
fpath2 = "validationData2.csv"

def read_data(fpath):
    train_df = pd.read_csv(fpath, header=0)
    # print(fpath + "finished reading. ")
    xl_length = len(train_df)
    # training data
    x = []
    # lables
    y_1 = []
    y_2 = []
    # dictionary: {building_floor:position}
    pos = {}

    idx = np.arange(xl_length)
    np.random.shuffle(idx)
    rows = np.asarray(train_df.iloc[:,:]).astype(float)
    sub_rows = np.asarray(train_df.iloc[:,0:520]).astype(float)

    # print("data finished processing - stage 1.")

    for i in idx:
        row = rows[i]
        y_1.append(one_hot_conversion(row[523],3))
        y_2.append(one_hot_conversion(row[522],5))

        row = (sub_rows[i] + 110) * 255 / 110
        x.append(row)

    # print("data finished processing - stage 2.")
    x = np.array(x)
    y_1 = np.array(y_1)
    y_2 = np.array(y_2)

    return x, y_1, y_2

x, y1, y2 = read_data(fpath1)
train_val_split = int(0.9 * len(x))  # mask index array
# train
x_train = np.array(x[:train_val_split])
y1_train = np.array(y1[:train_val_split])
y2_train = np.array(y2[:train_val_split])
# validation
x_val = np.array(x[train_val_split:])
y1_val = np.array(y1[train_val_split:])
y2_val = np.array(y2[train_val_split:])
# test
x_test, y1_test, y2_test = read_data(fpath2)
x_test = np.array(x_test)
y1_test = np.array(y1_test)
y2_test = np.array(y2_test)

signals_input = Input(shape=(520,))
x = layers.Dense(256, activation='relu')(signals_input)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(8, activation='relu')(x)
x = layers.Dropout(0.2)(x)


building_prediction = layers.Dense(3, activation='softmax', name='building')(x)
floor_prediction = layers.Dense(5, activation='softmax', name='floor')(x)

model = Model(signals_input, [building_prediction, floor_prediction])

model.summary()

model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'categorical_crossentropy'])

history = model.fit(x_train, [y1_train, y2_train], epochs=20, batch_size=64, validation_data=(x_val,[y1_val, y2_val]))

# test_loss, test_acc = model.evaluate(x_test,[y1_test, y2_test])

# print('Test accuracy:', test_acc)

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# plt.clf()

# acc = history.history['acc']
# val_acc = history.history['val_acc']

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()

predictions1,predictions2 = model.predict(x_test)

b = 0
b_f = 0

print(predictions1)

for i in range(1, len(x_test)):
    sub1 = np.argmax(predictions1[i])
    sub2 = np.argmax(predictions2[i])

    sub4 = np.argmax(y1_test[i])
    sub5 = np.argmax(y2_test[i])

    if (sub1 == sub4) and (sub2 == sub5):
        b_f += 1
        b += 1
    else:
        if (sub1 == sub4):
            b += 1

s = len(x_test)
a1 = 100 * b / s
a2 = 100 * b_f / s

print("For ", s, " test data:")
print("building accuracy: ", a1, "%")
print("building + floor prediction accuracy: ", a2, "%")

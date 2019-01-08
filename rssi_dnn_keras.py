from tensorflow import keras 
import numpy as np 
import rssi_data as data
import tensorflow as tf 

x,y = data.read_datasets()
train_val_split = int(0.7 * len(x))  # mask index array
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
x_val = np.array(x[train_val_split:])
y_val = np.array(y[train_val_split:])


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
model.add(keras.layers.Dense(118, activation='sigmoid'))



model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_val,y_val)

print('Test accuracy:', test_acc)

predictions = model.predict(x_val)

diff_building = 0
diff_floor = 0
diff_place = 0

for i in range(1,len(x_val)):
    sub1 = np.argmax(predictions[-i][:3])
    sub2 = np.argmax(predictions[-i][3:8])
    sub3 = np.argmax(predictions[-i][8:118])

    sub4 = np.argmax(y_val[-i][:3])
    sub5 = np.argmax(y_val[-i][3:8])
    sub6 = np.argmax(y_val[-i][8:118])

    if (sub1 != sub4):
        diff_building += 1
    else:
        if (sub2 != sub5):
            diff_floor += 1
        else:
            if (sub3 != sub6):
                diff_place += 1

s = len(x_val)
a1 = 100 * (s - diff_building) / s
a2 = 100 * (s - diff_floor) / s
a3 = 100 * (s - diff_place) / s

print("For ",s," test data:")
print("building accuracy: ", a1,"%")
print("building + floor prediction accuracy: ", a2,"%")
print("building + floor + place accuracy: ", a3,"%")
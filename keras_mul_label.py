from tensorflow import keras 
import numpy as np 
import rssi_data as data
import tensorflow as tf 

x,y = data.read_datasets()
train_val_split = int(0.7 * len(x))  # mask index array
val_test_split = int(0.8 * len(x))
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
model.add(keras.layers.Dense(118, activation='sigmoid'))



model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20,batch_size=512,validation_data=(x_val,y_val))

test_loss, test_acc = model.evaluate(x_test,y_test)

print('Test accuracy:', test_acc)

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
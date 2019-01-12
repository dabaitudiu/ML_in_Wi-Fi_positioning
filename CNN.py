from keras import layers 
from keras import models 
import rssi_data as train_data 
import read_test_data as test_data 
import tensorflow as tf 
import numpy as np 


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(23,23,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(118, activation='sigmoid'))

model.summary()

x, y = train_data.cnn_read()
train_val_split = int(0.9 * len(x))  # mask index array

# train
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
# validation
x_val = np.array(x[train_val_split:])
y_val = np.array(y[train_val_split:])
# test
x_test, y_test = test_data.cnn_read()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20,batch_size=512,validation_data=(x_val,y_val))

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


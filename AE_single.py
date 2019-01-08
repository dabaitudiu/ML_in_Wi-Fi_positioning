from tensorflow import keras 
import numpy as np 
import rssi_data as data
import tensorflow as tf 
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
# from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle

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


# Single fully-connected neural layer as encoder and decoder

use_regularizer = True
my_regularizer = None
my_epochs = 100
features_path = 'simple_autoe_features.pickle'
labels_path = 'simple_autoe_labels.pickle'

if use_regularizer:
    # add a sparsity constraint on the encoded representations
    # note use of 10e-5 leads to blurred results
    my_regularizer = regularizers.l1(10e-8)
    # and a larger number of epochs as the added regularization the model
    # is less likely to overfit and can be trained longer
    my_epochs = 100
    features_path = 'sparse_autoe_features.pickle'
    labels_path = 'sparse_autoe_labels.pickle'

# this is the size of our encoded representations
encoding_dim = 256   # 32 floats -> compression factor 24.5, assuming the input is 784 floats

# this is our input placeholder; 
input_img = Input(shape=(520, ))

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(520, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# Separate Decoder model

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Train to reconstruct MNIST digits

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# prepare input data
x_test = x_val

# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
print(x_train.shape)
print(x_test.shape)

# Train autoencoder for 50 epochs

autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test),
                verbose=2)

# after 50/100 epochs the autoencoder seems to reach a stable train/test lost value

# Visualize the reconstructed encoded representations

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
print(encoded_imgs.shape)
decoded_imgs = decoder.predict(encoded_imgs)

# # save latent space features 32-d vector
# pickle.dump(encoded_imgs, open(features_path, 'wb'))
# pickle.dump(y_test, open(labels_path, 'wb'))

# n = 10  # how many digits we will display
# plt.figure(figsize=(10, 2), dpi=100)
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.set_axis_off()

#     # display reconstruction
#     ax = plt.subplot(2, n, i + n + 1)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.set_axis_off()

# plt.show()

x_train_compressed = encoder.predict(x_train)
x_val_compressed = encoder.predict(x_val)
print(x_train_compressed.shape)
print(x_val_compressed.shape)
print(x_train_compressed[0])
print(type(x_train_compressed))


model = keras.Sequential()
model.add(keras.layers.Dense(256, activation='relu', input_shape=(256,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(118, activation='sigmoid'))

# x_train_compressed = encoder.predict(x_train)
# x_val_compressed = encoder.predict(x_val)

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_compressed, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_val_compressed,y_val)

print('Test accuracy:', test_acc)

predictions = model.predict(x_val_compressed)

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
# K.clear_session()
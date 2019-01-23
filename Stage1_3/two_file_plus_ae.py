from tensorflow import keras 
import numpy as np 
import rssi_data as data
import read_test_data as test_data 
import tensorflow as tf 
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
# from keras import backend as K
import matplotlib.pyplot as plt
import pickle

x,y = data.read_datasets()
train_val_split = int(0.9 * len(x))  # mask index array
# train
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
# validation
x_val = np.array(x[train_val_split:])
y_val = np.array(y[train_val_split:])
# test
x_test, y_test = test_data.read_datasets()
x_test = np.array(x_test)
y_test = np.array(y_test)


print("data shapes:")
print(x.shape)
print(y.shape)
print("train shapes:")
print(x_train.shape)
print(y_train.shape)


### Build SAE
print("\n Part 1 : Build Stacked Autoencoder:")

# Single fully-connected neural layer as encoder and decoder

use_regularizer = True
my_regularizer = None
my_epochs = 20
features_path = 'simple_autoe_features.pickle'
labels_path = 'simple_autoe_labels.pickle'

if use_regularizer:
    # add a sparsity constraint on the encoded representations
    # note use of 10e-5 leads to blurred results
    my_regularizer = regularizers.l1(10e-8)
    # and a larger number of epochs as the added regularization the model
    # is less likely to overfit and can be trained longer
    my_epochs = 20
    features_path = 'sparse_autoe_features.pickle'
    labels_path = 'sparse_autoe_labels.pickle'

# this is the size of our encoded representations
encoding_dim = 256   # 32 floats -> compression factor 24.5, assuming the input is 784 floats

# this is our input placeholder; 
input_img = Input(shape=(520, ))

# # "encoded" is the encoded representation of the inputs
# encoded = Dense(encoding_dim * 2, activation='relu', activity_regularizer=my_regularizer)(input_img)
# encoded = Dense(encoding_dim, activation='relu')(encoded)

# # "decoded" is the lossy reconstruction of the input
# decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
# decoded = Dense(520, activation='sigmoid')(decoded)

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
# decoder_layer1 = autoencoder.layers[-2]
# decoder_layer2 = autoencoder.layers[-1]
decoder_layer = autoencoder.layers[-1]

# create the decoder model
# decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))
decoder = Model(encoded_input, decoder_layer(encoded_input))


# Train to reconstruct MNIST digits

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(x_val, x_val),
                verbose=2)

x_train_c = encoder.predict(x_train)
x_val_c = encoder.predict(x_val)
x_test_c = encoder.predict(x_test)

model = keras.Sequential()
model.add(keras.layers.Dense(256, activation='relu', input_shape=(256,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(118, activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_c, y_train, epochs=20,batch_size=512,validation_data=(x_val_c,y_val))

test_loss, test_acc = model.evaluate(x_test_c,y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(x_test_c)

b = 0
b_f = 0
b_f_p = 0

for i in range(1,len(x_test_c)):
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

s = len(x_test_c)
a1 = 100 * b / s
a2 = 100 * b_f / s
a3 = 100 * b_f_p / s

print("For ",s," test data:")
print("building accuracy: ", a1,"%")
print("building + floor prediction accuracy: ", a2,"%")
print("building + floor + place accuracy: ", a3,"%")

## ML in Indoor Positioning

#### Stage 1
KNN + Gradient Descent

#### Stage 2
Feed Forward, single layer

#### Stage 3
keras 256_128_118, dropout = 0.2, multi-label(,118), epoch = 20, loss = binary_cross_entropy
```python
building accuracy:  100.0 %
building + floor prediction accuracy:  87.82608695652173 %
building + floor + place accuracy:  43.55555555555556 %
```

#### Stage 4
Autoencoder:
1. AE - 256 (single) (finished.)
```python
Without AE:
Test accuracy: 0.9893886969153687
For  5981  test data:
building accuracy:  99.73248620632002 %
building + floor prediction accuracy:  86.94198294599565 %
building + floor + place accuracy:  36.81658585520816 %

# --------------------------------------------------------------
With AE:
Test accuracy: 0.9888630269303647
For  5981  test data:
building accuracy:  99.08042133422505 %
building + floor prediction accuracy:  85.45393746865072 %
building + floor + place accuracy:  30.245778297943488 %
```
**2019-Jan-9**: 
1. correct prediction calucaltions, which were written wrongly.
```python
b = 0
b_f = 0
b_f_p = 0

for i in range(1,len(x_val)):
    sub1 = np.argmax(predictions[-i][:3])
    sub2 = np.argmax(predictions[-i][3:8])
    sub3 = np.argmax(predictions[-i][8:118])

    sub4 = np.argmax(y_val[-i][:3])
    sub5 = np.argmax(y_val[-i][3:8])
    sub6 = np.argmax(y_val[-i][8:118])

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
```
2. with shuffle on input data, level 2 reaches 99% accuracy. Meanwhile, separate multiclass training (manual group buildings) does not show significant enhancement on prediction accuracy. [Model: 520-256-128-64-32-16-5] , dropout=0.2This perhaps because building + floor prediction has already been very high. Its effect on Point prediction has not been tested, which will be tested later. 
```python

idx = np.arange(1, xl_length)
np.random.shuffle(idx)

for i in idx:
    row = np.array(xl_table.row_values(i))
...

model = keras.Sequential()
model.add(keras.layers.Dense(256, activation='relu', input_shape=(520,)))
model.add(keras.layers.Dropout(0.2))
...
model.add(keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=11,batch_size=512,validation_data=(x_val,y_val))
        
# ------------------------------------------------------------

For  5981  test data:
building accuracy:  99.83280387895 %
building + floor prediction accuracy:  99.31449590369503 %
building + floor + place accuracy:  58.68583848854706 %

# ------------------------------------------------------------

Separate multiclass classification: 
Floor prediction accuracy at Building 0: 99.03846153846153%
Floor prediction accuracy at Building 1: 98.84615384615384%
Floor prediction accuracy at Building 2: 99.57894736842105%
```
3. file: data_statistics.py is created, in which we can find the signal input distributions. There are too much 0s after regularization, which means that this is a sparse input. By far, I have little idea about handling sparse input, whether to use sparse autoencoder or other methods will be fulfilled in the following days.

```python
[1.0007061e+07 4.9434000e+04 1.4171300e+05 8.5343000e+04 4.7117000e+04
 2.2588000e+04 9.8640000e+03 1.7560000e+03 6.6000000e+01 4.8700000e+02
 1.1400000e+02 1.3700000e+02]
 ```

4. As the building+floor prediction has been very positive, the only problem left is to enhance the accuracy prediction on **Point**. Besides Neural Networks, there are also traditional ways to handle this classification problem. For example, Random Forest. These will also be tested in the following days. 

**2019-Jan-10**: 
1. manual_floor.py Updated. With which can testify multiclass classification with hierarchical structures that are formed manually. First thinking is to adjust epochs and other parameters according to the validation graph. 

**2019-Jan-11**:
modify rssi_dnn_keras.py. Rectify errors, optimize codes, correctly split data into train, val, test sets.

**2019-Jan-12**:
modify rssi_data.py, read_test_data.py, created toImageTest.py, CNN.py
Implemented CNN, whereas the accuracy is not high.
```python
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 21, 21, 32)        320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 10, 10, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 64)          18496
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 2, 2, 64)          36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792
_________________________________________________________________
dense_2 (Dense)              (None, 118)               30326
=================================================================
Total params: 151,862
Trainable params: 151,862
Non-trainable params: 0
_________________________________________________________________

For  1111  test data:
building accuracy:  98.37983798379838 %
building + floor prediction accuracy:  70.83708370837084 %
```
Uploaded Optimized_Manual_Buildings.py, Manual_Buildings_with_CNN.py
```python
For 536 test data:
floor accuracy:  94.21641791044776 %

For 307 test data:
floor accuracy:  82.41042345276873 %

For 268 test data:
floor accuracy:  92.91044776119404 %
```
No significant improvements happened. 

**2019-Jan-15**:
created count_points_at_floors.py. Recount the points at different floors, which proves that the number of labels is not equal to 110.
```python
2-3  :  91
2-0  :  44
1-1  :  38
1-3  :  30
0-2  :  68
2-2  :  57
0-3  :  68
0-1  :  66
1-2  :  45
0-0  :  54
1-0  :  49
2-1  :  60
2-4  :  65
```
tensorflow-gpu终于work了。。安装问题及总结归在**bug_collection** repo了。

**2019-Jan-15**:
连续鼓捣6个多小时终于连file带gpu配置给弄完了。。
created **VGG16_test.py**. But just ran 10 epochs due to the speed limit of my GPU.
```python
For  1111  test data:
building accuracy:  98.28982898289829 %
building + floor prediction accuracy:  87.3987398739874 %
building + floor + place accuracy:  1.3501350135013501 %
```
Tomorrow will test (1) 20 epochs (2) self-trained weights 

### Stage 5 
2. AE - denoised
3. AE - stacked
4. AE - CNN
5. CNN
6. AE + CNN
7. AE_CNN + CNN

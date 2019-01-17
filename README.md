## ML in Indoor Positioning

#### Stage 1
KNN + Gradient Descent

#### Stage 2
Feed Forward, single layer

#### Stage 3
- 256_128_118
- multi-label
- [keras_multi_label.py](https://github.com/dabaitudiu/FYP/blob/master/keras_mul_label.py)
- [mul_use_two_files.py](https://github.com/dabaitudiu/FYP/blob/master/mul_use_two_file.py)

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

**2019-Jan-16**:
连续鼓捣6个多小时终于连file带gpu配置给弄完了。。
created **VGG16_test.py**. But just ran 10 epochs due to the speed limit of my GPU.
```python
For  1111  test data:
building accuracy:  98.28982898289829 %
building + floor prediction accuracy:  87.3987398739874 %
building + floor + place accuracy:  1.3501350135013501 %
```
- 成功在cloud上跑程序。 2080Ti诚不我欺。推荐[极客云](http://www.jikecloud.net/register.html?iid=nxjgaUz3gadPt2hzEBR8ig==).
- 20，30 epochs都试了，VGG16 with pretrained_weights最好记录是87%， None weights最好记录89%
- 不知道Jang & Hong的paper是怎么到的95%的accuracy的，感觉有点玄幻。这个问题真的不适合CNN。强行转CNN还行？
- 之前为了满足VGG16最小输入条件32_32，直接拼接了两次数组，今天用了PIL库的resize函数试了一下，效果特别差。 70%多
- Created [Multi-Head.py](https://github.com/dabaitudiu/FYP/blob/master/Multi_Head.py). 效果很好，能达到91%。
- Optimized Reading function. Used pandas instead of xlrd, speed increases significantly.

**2019-Jan-17**:
**发现重要问题**： 
- Kim的paper中Referenced Points的分类根本就不对！他按Building-Floor-Referenced_Points组成label, Building和Floor没什么问题，但是！不同的Building和Floor组合可能有相同的Referenced_Points值！比如2楼4层106号点位：2_4_106, 也可能有3楼3层106号点位: 3_3_106, 但是这两个106不是一个东西！此外，我之前的处理方法更有问题，我是按key=building_floor对点位进行了分组，但是label还是用的max(len(点位)), 这肯定就错了，因为label的值代表的根本就不是一个东西。怪不得accuracy这么低。
- 删掉label重写了一遍CNN:[CNN_BF.py](https://github.com/dabaitudiu/FYP/blob/master/CNN_BF.py)， 果然accuracy提高到91%以上。

**2019-Jan-18**:
[CNN based Indoor Localization using RSS Time-Series](https://www.researchgate.net/publication/325678644_CNN_based_Indoor_Localization_using_RSS_Time-Series)
今天读完这篇paper我觉得我可以放弃继续fyp了。Building,Floor predication全100%， 把我发现之前paper的问题都给总结了， 我是真滴佛了。。。
这个paper也采用了我一直认为该用的分层训练，细化到点之后，每个点大概20多个samples，做成矩阵放进CNN。。。双100%，，，我是真滴佛了。。
Created Group_BF.py. 其实没什么。。但我这脑力大概是我写过最绕的程序了。。


### Stage 5 
2. AE - denoised
3. AE - stacked
4. AE - CNN
5. CNN
6. AE + CNN
7. AE_CNN + CNN

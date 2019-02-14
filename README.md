## ML in Indoor Positioning

#### Stage 1-3： Simple classification trials
KNN + Gradient Descent, (256, 128, 128) NN, 多标签分类
- [keras_multi_label.py](https://github.com/dabaitudiu/FYP/blob/master/Stage1_3/keras_mul_label.py)
- [mul_use_two_files.py](https://github.com/dabaitudiu/FYP/blob/master/Stage4/mul_use_two_file.py)

#### Stage 4： Autoencoder for reduce dimensions
- [AE - 256](https://github.com/dabaitudiu/FYP/blob/master/Stage4/AE_single.py)
- [AE - 256-128-64-128-256](https://github.com/dabaitudiu/FYP/blob/master/Stage4/AE_64to118.py)

#### Stage 5：CNN, VGG16, InceptionV3, Multi-Head, Random Forest, XGBoost.
**2019-Jan-9**: 
- correct prediction calucaltions, which were written wrongly.

- with [shuffle](https://github.com/dabaitudiu/FYP/blob/master/Stage4/breakthrough_shuffle.py) on input data, predictions accuracies increased significantly. Meanwhile, separate multiclass training (manual group buildings) does not show significant enhancement on prediction accuracy. [Model: 520-256-128-64-32-16-5].This perhaps because building + floor prediction has already been very high. Its effect on Point prediction has not been tested, which will be tested later. 

- file: [data_statistics.py](https://github.com/dabaitudiu/FYP/blob/master/data_statistics) is created, in which we can find the signal input distributions. There are too much 0s after regularization, which means that this is a sparse input. By far, I have little idea about handling sparse input, whether to use sparse autoencoder or other methods will be fulfilled in the following days.

- As the building+floor prediction has been very positive, the only problem left is to enhance the accuracy prediction on **Point**. Besides Neural Networks, there are also traditional ways to handle this classification problem. For example, Random Forest. These will also be tested in the following days. 

**2019-Jan-10**: 
- [manual_floor.py](https://github.com/dabaitudiu/FYP/blob/master/Stage4/manual_floor.py) Updated. With which can testify multiclass classification with hierarchical structures that are formed manually. First thinking is to adjust epochs and other parameters according to the validation graph. 

**2019-Jan-11**:
- modify rssi_dnn_keras.py. Rectify errors, optimize codes, correctly split data into train, val, test sets.

**2019-Jan-12**:
- modify rssi_data.py, read_test_data.py, created toImageTest.py, CNN.py
- Implemented CNN, whereas the accuracy is not high.
- Uploaded Optimized_Manual_Buildings.py, Manual_Buildings_with_CNN.py

**2019-Jan-15**:
- created [count_points_at_floors.py](https://github.com/dabaitudiu/FYP/blob/master/count_points_at_floors.py). Recount the points at different floors, which proves that the number of labels is not equal to 110.
- tensorflow-gpu终于work了。。安装问题及总结归在**bug_collection** repo了。

**2019-Jan-16**:
- 连续鼓捣6个多小时终于连file带gpu配置给弄完了。。
- created **VGG16_test.py**. But just ran 10 epochs due to the speed limit of my GPU.
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
- 今天读完这篇paper我觉得我可以放弃继续fyp了。Building,Floor predication全100%， 把我发现之前paper的问题都给总结了， 我是真滴佛了。。。这个paper也采用了我一直认为该用的分层训练，细化到点之后，每个点大概20多个samples，做成矩阵放进CNN。。。双100%，，，
- Created Group_BF.py. 其实没什么。。但我这脑力大概是我写过最绕的程序了。。

**2019-Jan-19 - 2019-Jan-22**:
- 集合并优化之前所有代码，删除之前错误和冗余代码，创建[Optimized_Manual_Buildings_2.py](https://github.com/dabaitudiu/FYP/blob/master/Optimized_Manual_Buildings_2.py)
- 测试了一下decision trees 和 random forest 的效果，decision trees 结果大概为[80%,80%,70%], random forest 大概能多了5%。
- data preprocessing: 去掉方差很小的data (0.9*0.1) features剩余409，B1的概率能提升到89%
- update extract_B1_F1.py, 83%的准确率确实需要额外分析。同学推荐的KL-divergence。

**2019-Jan-23**:
- Created [Hyperopt_test.py](https://github.com/dabaitudiu/FYP/blob/master/Hyperopt_test.py)用了一下Hyperopt调参，不过没什么improvement.
- Created [xgboost_test.py](https://github.com/dabaitudiu/FYP/blob/master/xgboost_test.py)天池之前有一个室内定位的比赛，看了几个总结贴，大部分都用了xgboost, 我也试了一下。感觉确实还可以， Building 0 96%, Building 2 93%，不过Building 1仍然只有77%左右，调参也很慢，用sci-kit的grid-search根本卡到不能运行。
- Created [model_ensemble_1.py](https://github.com/dabaitudiu/FYP/blob/master/model_ensemble_1.py) 试了一下模型融合，NN和XGB各有错误，共同错误仅仅有一个。咨询了一下同学，建议添加更多模型，尽管准确率低。之后用模型做regression或voting。
- 重新验证了一下Kim_s.py这个文件，一直好奇他是怎么实现floor92%以上的准确率的，毕竟有一个是83%. 数据处理的太不一样，实在是搞不下去，后来对自己的结果做了一下加权，发现也在92%以上，所以估计他这个模型还算靠谱。不过他瞎标label是肯定错了。
- LightGBM还没测试，估计之后几天也就是模型融合了。
- Uploaded [Random_Forest_test.py](https://github.com/dabaitudiu/FYP/blob/master/Random_Forest_test.py) 上传随机森林test，准备调参voting.

**2019-Jan-26**:
- Created [feature_selection.py](https://github.com/dabaitudiu/FYP/blob/master/feature_selection.py)
- 用了一下sklearn的特征提取，200feautre有用，之后random forest在Building 0 稳定97%以上准确率。不过后来也试了一下KNN, N=10时候竟然也到了97%
- 重新看了一遍CNN那个paper，觉得非常有问题。
- (1) **Ambiguous test data**: They first claimed they will use the 1111 testing records, but in the later sections of the paper, they redefined the testing set again by generating it from the 20% of the training set. This is really worth considering, because if I do the same thing as him (rather than using the 1111 test data), I can also reach 100% accuracy. <br/> (2) **One-sided assertion**: As they say, they only select building #2 for testing, just because it will be the same for the other buildings intuitively. This is really not rigorous. Even if their algorithm really achieves higher accuracy, they did not test them on Building #1 (where I get 80% accuracy).<br/> (3) **Data format issue for testing**: In their work, they captured 10 consistent readings from a place and put the 10_520 input to a CNN model. It is fine with the training set, however, for the test data (which only has very few samples), it’s hard to form 10 consistent readings following their specifications. In other words, for testing data, I can only find some 3_520, or 7_520 matrix, rather than their 10 * 520 readings. This also supports my suspect that their test data is generated from the train data.<br/> (4)**In conclusion**, I think this paper is really not very rigorous, or maybe I understand wrongly. Secondly, after feature selection, I think with current classification methods, ensemble models like random forests and gradient boosting has already reached the bottleneck for the prediction accuracy, there should be some problems with the test set for Building #1. Finally, if I split the train set to train and test set, I can also reach 100% prediction accuracy already. 

**2019-Jan-29**:
- 能想到的现有model都测试了一遍，陷入迷茫，在思考有没有什么新的方向。决定先算一下自己目前meter级的准确度.
- created [Meters_test.py](https://github.com/dabaitudiu/FYP/blob/master/Meters_test.py) 本来想在一个文件内解决，脑子不够用，总搞混，先把坐标信息提取到一个csv里了。

**2019-Feb-11**:
- [Point_Prediction.py](https://github.com/dabaitudiu/FYP/blob/master/Point_Prediction.py) 和 final_data.csv是年前搞的，主要就是把point label和经纬度match一下
- 好久没做fyp了。。再不做估计就凉了。今天整合了一下所有的code，[BFP_Building.py](https://github.com/dabaitudiu/FYP/blob/master/BFP_Building.py), [BFP_Floor.py](https://github.com/dabaitudiu/FYP/blob/master/BFP_Floor.py), [BFP_Point.py](https://github.com/dabaitudiu/FYP/blob/master/BFP_Point.py)算了一下point的meters error，12.5m，哭了。等会可能要trim一下再看看。
- 之前想到一个思路是同时算classification和regression， 如果差距太大就是错误，要重算
- 对于Building1特别不准，昨天看了一个NLP比赛的总结，对于这种情况可以用boosting，不过啥是boosting，具体能不能用还得之后再看。
- Regression写完了 [data_trim.py](https://github.com/dabaitudiu/FYP/blob/master/data_trim.py), 结果惨不忍睹，根本没有参考价值...
```python

Total Average error:  12.094245732001768

A ( < 50) number  1072
A Average error:  9.90055803326563

B ( < 100) number  1105
Average error:  11.528668641755623
```

**2019-Feb-14**:
- 终于做了一下模型融合[Model_Assembling.py](https://github.com/dabaitudiu/FYP/blob/master/Model_Assembling.py)，然而效果并不好
- Error Distribution:
<img src="https://github.com/dabaitudiu/FYP/blob/master/meters_distribution.png" width = "300" height = "200" alt="fig1" align="middle"/>


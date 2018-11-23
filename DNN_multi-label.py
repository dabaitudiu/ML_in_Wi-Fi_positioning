# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import xlrd
import csv

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

myInput = xlrd.open_workbook("datav2.xlsx")
print("open finished.")


miTable = myInput.sheets()[0]
length = miTable.nrows
pos = {}
y = []
x = []

def myOneHotCode(n,i):
    a = np.zeros(n)
    a[int(i)] = 1
    return a

# csvFile = open("manipulated_indoor_new.csv", "w", newline='')
# writer = csv.writer(csvFile)

coordinates = {}

for i in range(0,length):
    row = miTable.row_values(i)
    row = np.array(row)

    # treat first 520 elements as inputs
    # for j in range(0,520):
    #     if row[j] == 100:
    #         row[j] = -110
    # writer.writerow(row)
    x.append((row[:520] + 110) / 110)

    # modifying row[524] as appearance orders
    myKey = str(int(row[523])) + "-" + str(int(row[522]))
    if myKey not in pos.keys():
        pos[myKey] = []
    if row[524] not in pos[myKey]:
        pos[myKey].append(row[524])
    row[524] = pos[myKey].index(row[524])
    coordinate_key = str(int(row[523]))+str(int(row[522]))+str(int(row[524]))
    if coordinate_key not in coordinates:
        coordinates[coordinate_key] = []
    coordinates[coordinate_key].append([row[520],row[521]])

    # one-hot code generation
    a = myOneHotCode(3,row[523])
    b = myOneHotCode(5,row[522])
    c = myOneHotCode(110,row[524])
    con = np.append(a,b)
    con = np.append(con,c)
    y.append(con)


train_val_split = int(0.7 * len(x)) + 1 # mask index array
x_train = np.array(x[:train_val_split])
y_train = np.array(y[:train_val_split])
x_val = np.array(x[train_val_split:])
y_val = np.array(y[train_val_split:])

print("train shapes:")
print(x_train.shape)
print(y_train.shape)


model = keras.Sequential([
    # keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu', input_shape=(520,)),
    keras.layers.Dense(118, activation='sigmoid')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

est_loss, test_acc = model.evaluate(x_val,y_val)

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

    # combined_result = str(sub1) + str(sub2) + str(sub3)
    # sumx = 0
    # sumy = 0
    # predictx = 0
    # predicty = 0
    # if combined_result in coordinates:
    #     coordinate_lists = coordinates[combined_result]
    #     for single_coordinate in coordinate_lists:
    #         sumx += coordinate_lists[single_coordinate][0]
    #         sumy += coordinate_lists[single_coordinate][1]
    #     predictx = sumx / len(coordinate_lists)
    #     predicty = sumy / len(coordinate_lists)
    # if i < 10:
    #     print()



s = len(x_val)
a1 = 100 * (s - diff_building) / s
a2 = 100 * (s - diff_floor) / s
a3 = 100 * (s - diff_place) / s

print("For ",s," test data:")
print("level 1 prediction accuracy: ", a1,"%")
print("level 2 prediction accuracy: ", a2,"%")
print("level 3 prediction accuracy: ", a3,"%")
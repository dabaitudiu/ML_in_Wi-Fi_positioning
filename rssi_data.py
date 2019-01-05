import xlrd 
import numpy as np 

fpath = "MNIST_data/test/train.xlsx"

def one_hot_conversion(a,b,c):
    m = [0] * 3
    n = [0] * 5
    q = [0] * 110
    m[int(float(a)-1)] = 1
    n[int(float(b)-1)] = 1
    q[int(float(c)-1)] = 1
    tmp = np.append(m,n)
    tmp = np.append(tmp,q)
    return tmp

# training data
x = []
# lables
y_ = []
# dictionary: {building_floor:position}
pos = {}

def read_datasets(filepath):
    xl_book = xlrd.open_workbook(filepath)
    xl_table = xl_book.sheets()[0] 
    rows = [np.array(xl_table.row_values(i)) for i in range(1,xl_table.nrows)]
    
    for i in range (len(rows)):
        row = rows[i]

        # modification on row[524] to make it range from 0 to 110 rather random number.
        myKey = str(row[523]) + "-" + str(row[522])
        if myKey not in pos.keys():
            pos[myKey] = []
        if row[524] not in pos[myKey]:
            pos[myKey].append(row[524])
        row[524] = pos[myKey].index(row[524])

        # append training data and lables
        x.append(row[:520])
        y_.append(one_hot_conversion(row[523],row[522],row[524]))
    return x, y_

def next_batch(num):
    idx = np.arange(0, len(x))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [x[i] for i in idx]
    labels_shuffle = [labels [i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
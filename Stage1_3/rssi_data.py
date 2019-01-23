import xlrd 
import numpy as np 


fpath = "MNIST_data/test/datav2.xlsx"


def one_hot_conversion(a,b,c):
    m = np.zeros(3)
    n = np.zeros(5)
    q = np.zeros(110)
    m[int(float(a))] = 1
    n[int(float(b))] = 1
    q[int(float(c))] = 1
    tmp = np.append(m,n)
    tmp = np.append(tmp,q)
    return tmp



def read_datasets():
    filepath = fpath
    xl_book = xlrd.open_workbook(filepath)
    xl_table = xl_book.sheets()[0] 
    xl_length = xl_table.nrows
    rows = [np.array(xl_table.row_values(i)) for i in range(1,xl_table.nrows)]

    # training data
    x = []
    # lables
    y_ = []
    # dictionary: {building_floor:position}
    pos = {}

    idx = np.arange(1, xl_length)
    np.random.shuffle(idx)

    for i in idx:
        row = np.array(xl_table.row_values(i))
        # modification on row[524] to make it range from 0 to 110 rather random number.
        myKey = str(row[523]) + "-" + str(row[522])
        if myKey not in pos.keys():
            pos[myKey] = []
        if row[524] not in pos[myKey]:
            pos[myKey].append(row[524])
        row[524] = pos[myKey].index(row[524])

        # append training data and lables
        x.append((row[:520].astype(np.float64)+110)/110)
        y_.append(one_hot_conversion(row[523],row[522],row[524]))

    x = np.array(x)
    print(x.shape)
    y_ = np.array(y_)
    return x, y_

def next_batch(num,x,y_):
    idx = np.arange(0, len(x))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [x[i] for i in idx]
    labels_shuffle = [y_[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def read_datasets2():
    filepath = fpath
    xl_book = xlrd.open_workbook(filepath)
    xl_table = xl_book.sheets()[0] 
    xl_length = xl_table.nrows
    rows = [np.array(xl_table.row_values(i)) for i in range(1,xl_table.nrows)]

    # training data
    x = []
    # lables
    y_ = []
    # dictionary: {building_floor:position}
    pos = {}

    for i in range (1,xl_length):
        row = np.array(xl_table.row_values(i))
        # append training data and lables
        x.append((row[:520].astype(np.float64)+110)/80)
        n = np.zeros(5)
        n[int(float(row[522]))] = 1
        y_.append(n)

    x = np.array(x)
    y_ = np.array(y_)
    return x, y_

def cnn_read():
    filepath = fpath
    xl_book = xlrd.open_workbook(filepath)
    xl_table = xl_book.sheets()[0] 
    xl_length = xl_table.nrows
    rows = [np.array(xl_table.row_values(i)) for i in range(1,xl_table.nrows)]

    # training data
    x = []
    # lables
    y_ = []
    # dictionary: {building_floor:position}
    pos = {}

    idx = np.arange(1, xl_length)
    np.random.shuffle(idx)

    for i in idx:
        row = np.array(xl_table.row_values(i))
        # modification on row[524] to make it range from 0 to 110 rather random number.
        myKey = str(row[523]) + "-" + str(row[522])
        if myKey not in pos.keys():
            pos[myKey] = []
        if row[524] not in pos[myKey]:
            pos[myKey].append(row[524])
        row[524] = pos[myKey].index(row[524])

        # append training data and lables
        sub_x = (row[:520].astype(np.float64)+110)/110
        # print("sub_x previous shape: ", sub_x.shape)
        ax = [np.zeros(9)]
        sub_x = np.append(sub_x, ax)
        # print("sub_x medium shape: ", sub_x.shape)
        sub_x = sub_x.reshape(23,23,1)
        # print("sub_x modified shape: ", sub_x.shape)
        x.append(sub_x)
        y_.append(one_hot_conversion(row[523],row[522],row[524]))

    x = np.array(x)
    print(x.shape)
    y_ = np.array(y_)
    
    return x, y_

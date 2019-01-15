import xlrd 
import numpy as np 


fpath = "datav2.xlsx"


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
        myKey = str(int(float(row[523]))) + "-" + str(int(float(row[522])))
        if myKey not in pos.keys():
            pos[myKey] = []
        if row[524] not in pos[myKey]:
            pos[myKey].append(row[524])
        row[524] = pos[myKey].index(row[524])

        # append training data and lables
        x.append((row[:520].astype(np.float64)+110)/110)
        y_.append(one_hot_conversion(row[523],row[522],row[524]))

    for item in pos.keys():
        print(item, " : ", len(pos[item]))

read_datasets()

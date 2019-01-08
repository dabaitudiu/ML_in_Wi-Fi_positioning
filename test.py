import xlrd 
import numpy as np 

fpath = "MNIST_data/test/train_minor.xlsx"


filepath = fpath
xl_book = xlrd.open_workbook(filepath)
xl_table = xl_book.sheets()[0]
xl_length = xl_table.nrows 
# rows = [np.array(xl_table.row_values(i)) for i in range(1,xl_table.nrows)]

def one_hot_conversion(a,b,c):
    print("inputs are: ",a,b,c)
    m = np.zeros(3)
    n = np.zeros(5)
    q = np.zeros(110)
    m[int(float(a))] = 1
    n[int(float(b))] = 1
    q[int(float(c))] = 1
    tmp = np.append(m,n)
    tmp = np.append(tmp,q)
    return tmp

# training data
x = []
# lables
y_ = []
# dictionary: {building_floor:position}
pos = {}
count = np.zeros(10)

for i in range (1,xl_length):
    row = np.array(xl_table.row_values(i))
    # modification on row[524] to make it range from 0 to 110 rather random number.
    myKey = str(row[523]) + "-" + str(row[522])
    if myKey not in pos.keys():
        pos[myKey] = []
    if row[524] not in pos[myKey]:
        pos[myKey].append(row[524])
    row[524] = pos[myKey].index(row[524])

    # append training data and lables
    x.append((row[:520].astype(np.float64)+110)/80)
    # for item in x[-1]:
    #     count[int(item * 10)] += 1
    y_.append(one_hot_conversion(row[523],row[522],row[524]))

    if i > xl_length - 10:
        print("train data:")
        print(x[-1])
        print("train label:")
        print(row[523],row[522],row[524])
        print("train label one-hot:")
        print(y_[-1])
        print("--------------------------------------------") 
   

print(x[-1])

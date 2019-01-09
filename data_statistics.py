import xlrd 
import numpy as np 
import matplotlib.pyplot as plt 

fpath = "MNIST_data/test/train_3000.xlsx"

filepath = fpath
xl_book = xlrd.open_workbook(filepath)
xl_table = xl_book.sheets()[0]
xl_length = xl_table.nrows

s = np.zeros(12)


for i in range(1, xl_length):
    row = np.array(xl_table.row_values(i))
    signals = row[:520]
    for item in signals:
        ori = float(item)
        if ori > 0:
            ori = -ori
        item = (ori + 110) / 10
        item = int(item)
        try:
            s[item] += 1
        except IndexError:
            print("at row " + " item: " + str(item))
            

print(s)
print(sum(s))

x = np.arange(1,12)
plt.figure()
plt.title('Signal Values Distribution')
plt.bar(x,s[1:])
plt.xlabel('Signal Strength Groups')
plt.ylabel('Number of Signals')
plt.show()

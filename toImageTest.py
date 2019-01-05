import numpy as np 
import scipy.misc 
import os 
import xlrd 


save_dir = 'MNIST_data/test/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)


xl_book = xlrd.open_workbook("MNIST_data/test/train.xlsx")
xl_table = xl_book.sheets()[0] 

for i in range(1,20):
    row = np.array(xl_table.row_values(i)) + 110
    ax = [np.zeros(9)]
    concax = np.append(row[:520],ax)
    concax = concax.reshape(23,23)
    filename = save_dir + '_%d.jpg'%i
    scipy.misc.toimage(concax, cmin=0.0, cmax=1.0).save(filename)


# print(row)
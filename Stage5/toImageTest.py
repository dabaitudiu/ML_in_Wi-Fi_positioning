import numpy as np 
import scipy.misc 
import os 
import xlrd 
import matplotlib.pyplot as plt 


save_dir = 'MNIST_data/images/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)


xl_book = xlrd.open_workbook("MNIST_data/test/train.xlsx")
xl_table = xl_book.sheets()[0] 
x_test = []

for i in range(1,110):
    # make it range from 0 ~ 255 
    row = (np.array(xl_table.row_values(i)[:520]).astype(np.float64) + 110) * 51 / 22
    ax = [np.zeros(9)]
    concax = np.append(row,ax)
    x_test.append(concax)
    concax = concax.reshape(23,23)

    filename = save_dir + '_%d.jpg'%i
    scipy.misc.toimage(concax, cmin=0.0, cmax=1.0).save(filename)


n = 10  # how many digits we will display
image_row = 10
plt.figure(figsize=(10, image_row), dpi=100)
plt.title("Image representation of signals from 520")
for i in range(n):
    for j in range(image_row):
        ax = plt.subplot(image_row, n, i + j * n + 1)
        plt.imshow(x_test[j * n + i].reshape(23, 23))
        plt.gray()
        ax.set_axis_off()

    

plt.show()
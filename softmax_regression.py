#coding:utf-8
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# read mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x stands for the input data
x = tf.placeholder(tf.float32, [None,784])

# W,b are paramenters in Softmax model, which turns a 784-d input to 10-d output
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# y stands for model's output
y = tf.nn.softmax(tf.matmul(x,W) + b)
# y_ is the actual label for images
y_ = tf.placeholder(tf.float32, [None,10])

# cross entropy between y and y_
corss_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
# optimize W and b using cross entropy. learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(corss_entropy)

# Create session
sess = tf.InteractiveSession()
# initialize all variables and allocate memory
tf.global_variables_initializer().run()

# gradient descent
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# results that are predicted correctly
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# computer prediction accuracy. accuracy & correct_prediction are both Tensor.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Run in session to get the value of tensor
print(sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

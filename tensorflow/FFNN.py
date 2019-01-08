#coding:utf-8
import tensorflow as tf 
import rssi_data as data

# read rssi data
signals,labels = data.read_datasets()

# x stands for the input data
x = tf.placeholder(tf.float32, [None,520])

# W,b are paramenters in Softmax model, which turns a 784-d input to 10-d output
W = tf.Variable(tf.zeros([520,118]))
b = tf.Variable(tf.zeros([118]))

# y stands for model's output
y = tf.nn.sigmoid(tf.matmul(x,W) + b)
# y_ is the actual label for images
y_ = tf.placeholder(tf.float32, [None,118])

# cross entropy between y and y_
corss_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
# optimize W and b using cross entropy. learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(corss_entropy)

# Create session
sess = tf.InteractiveSession()
# initialize all variables and allocate memory
tf.global_variables_initializer().run()

# gradient descent
for _ in range(1000):
    batch_xs, batch_ys = data.next_batch(100,signals,labels)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if (_ % 200 == 0):
        print(_)

# results that are predicted correctly
correct_prediction_building = tf.equal(tf.argmax(y[:,:3],1), tf.argmax(y_[:,:3],1))
correct_prediction_floor = tf.equal(tf.argmax(y[:,3:8],1), tf.argmax(y_[:,3:8],1))
correct_prediction_place = tf.equal(tf.argmax(y[:,8:],1), tf.argmax(y_[:,8:],1))

# computer prediction accuracy. accuracy & correct_prediction are both Tensor.
accuracy_building = tf.reduce_mean(tf.cast(correct_prediction_building, tf.float32))
accuracy_floor = tf.reduce_mean(tf.cast(correct_prediction_floor, tf.float32))
accuracy_place = tf.reduce_mean(tf.cast(correct_prediction_place, tf.float32))

# Run in session to get the value of tensor
print(sess.run(accuracy_building, feed_dict={x:signals,y_:labels}))
print(sess.run(accuracy_floor, feed_dict={x:signals,y_:labels}))
print(sess.run(accuracy_place, feed_dict={x:signals,y_:labels}))


m = y.eval(session=sess,feed_dict={x:signals,y_:labels}) 
n = y_.eval(session=sess,feed_dict={x:signals,y_:labels}) 
print(m[0])
print("-------------------")
print(n[0])

# with open('MNIST_data/test/record.txt', 'a+') as f:
#     f.write(m[:,:8])
#     f.write("   ")
#     f.write(n[:,:8])


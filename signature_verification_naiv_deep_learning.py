from SVC_convert import *

import numpy as np
import tensorflow as tf
import random

datas = createSignatureDataSet("SVC/024/")

train_set_size = 20

train_data = datas[:train_set_size]
test_data = datas[train_set_size:]

x = tf.placeholder(tf.float32, [None, 108000])
W = tf.Variable(tf.zeros([108000, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(10):
    batch_xs, batch_ys = getSignatureImagesAndLabels(get_random_batch(train_data,5))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
batch_xs, batch_ys = getSignatureImagesAndLabels(test_data)
print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
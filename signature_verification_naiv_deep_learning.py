def read_image(path):
    import cv2
    import numpy as np
    
    image = cv2.imread(path)[0:180,0:600]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("cropped",gray_image)
    #cv2.waitKey(0)
    
    #print path + ", " + str(gray_image.flatten().shape)
    
    label = np.array([0,1])
    
    if "e" in path:
        label = np.array([1,0])
    
    return gray_image.flatten(), label;
    
def get_random_batch(datas, count):
    from random import randint
    
    raw_data = []
    label = []
        
    for c in range(0,count):
        i = randint(0,len(datas[0])-1)
        raw_data.append(datas[0][i])
        label.append(datas[1][i])
        
    return raw_data, label
    
def createSignatureDataSet(signature_path):
    import numpy as np
    import glob
    
    image_files = glob.glob(signature_path + "*.png")
    
    raw_data = []
    label = []
    
    for image_path in image_files:
        signature = read_image(image_path)
        
        raw_data.append(signature[0])
        label.append(signature[1])

    return np.array(raw_data), np.array(label)


import numpy as np
import tensorflow as tf
import random
datas = createSignatureDataSet("SVC/024/")

train_set_size = 20

train_data = (datas[0][:train_set_size]), (datas[1][:train_set_size])
test_data = (datas[0][train_set_size:]), (datas[1][train_set_size:])

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
    batch_xs, batch_ys = get_random_batch(train_data,5)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_data[0], y_: test_data[1]}))
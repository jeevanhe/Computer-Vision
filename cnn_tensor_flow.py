# Experiments with Convolutional Neural Network in TensorFlow
# Background
# Convolutional neural network is a powerful learning mechanism that was applied successfully to
# many computer vision tasks. In this project we will run experiments using tensor
# ow. The experiments are closely related to the online example available at:
# https://www.tensorflow.org/get_started/mnist/pros.
# The above online example shows how to apply a convolutional neural net to learn how to classify
# data from the MNIST dataset. The code in the example takes about 30 minutes to train, and
# achieves over 99% accuracy.
# In this project you are asked to solve the exact same problem: creating a classier for the
# MNIST data. However, we put two constraints on the network being created that are intended to
# reduce the network accuracy and improve the training time:
# 1. The first layer in the network must be a 2 * 2 maxpool layer.
# 2. Training must be with only 1000 batches. (It is up to you to decide on the batch size.)

# Template for project/homework, learning with deep convolutional network
# The template program starts with a 2x2 max-pool to reduce input size. 
#(This is a very bad idea for anything but academic experiments...)
# The template shows how to use a convolutional layer, a fully connected layer, and dropout

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(X, W):
  return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(X):
  return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

orig_image = tf.reshape(X, [-1,28,28,1])
### For this assignment 2x2 max pool layer must be the first layer ###
h_pool0 = max_pool_2x2(orig_image)
### End of first max pool layer ###

### beginning of layer definitions

# Convolutional Layer

W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Densely Connected Layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool1_flat = tf.reshape(h_pool1, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

### end of layer definitions

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(1000):
  batch = mnist.train.next_batch(100)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        X:batch[0], Y: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))

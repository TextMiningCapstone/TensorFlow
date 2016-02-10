import tensorflow as tf
import numpy as np
import dataGen as gen

# N number of points per class
# D dimensionality
# K number of classes
def softMax(N, D, K):
    # Start a session
    sess = tf.InteractiveSession()
    # Generate data, total no. of data = N*K; data is [N*K, D]; label is [N*K, 1]
    [data, initLabel] = gen.genData(N, D, K)
    # Reform label, each class is represent by a vector
    label = np.zeros((N*K,K))
    for i in range(N*K):
        label[i][initLabel[i]] = 1
    # Build the computation graph by creating nodes for the input and target output classes
    x = tf.placeholder("float", shape=[None, D])
    y_ = tf.placeholder("float", shape=[None, K])
    # Define weights and Bias
    W = tf.Variable(tf.zeros([D,K]))
    b = tf.Variable(tf.zeros([K]))
    # Initialize all variables
    sess.run(tf.initialize_all_variables())
    # Define prediction function y
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    # Define cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # Train model
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    for i in range(5000):
      train_step.run(feed_dict={x: data, y_: label})
    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print accuracy.eval(feed_dict={x: data, y_: label})


N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
# total no. of data = N*K; data is [N*K, D]; label is [N*K, 1]
softMax(N, D, K)

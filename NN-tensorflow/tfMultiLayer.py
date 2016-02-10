from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import dataGen as gen

import tensorflow.python.platform
import tensorflow as tf

# The dataset has 3 classes, representing the digits 0 through 2.
NUM_CLASSES = 3
# The dimension is 2
DIMENSION = 2
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 5, 'Number of units in hidden layer 2.')
# flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
#                      'Must divide evenly into the dataset sizes.')
# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
# flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
#                      'for unit testing.')


def inference(data, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    data: Data placeholder.
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([DIMENSION, hidden1_units],
                            stddev=1.0 / math.sqrt(float(DIMENSION))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

# number of points per class
N = 100
[data, labels] = gen.genData(N, DIMENSION, NUM_CLASSES)
[test_data, test_labels] = gen.genData(N, DIMENSION, NUM_CLASSES)

# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    data_placeholder = tf.placeholder(tf.float32, shape=(None, DIMENSION))
    labels_placeholder = tf.placeholder(tf.int32, shape=(None))
    # Build a Graph that computes predictions from the inference model.
    logits = inference(data_placeholder, FLAGS.hidden1, FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()
    #
    # # Create a saver for writing training checkpoints.
    # saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)
    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = {data_placeholder: data, labels_placeholder: labels}

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

    #test_data_placeholder = tf.placeholder(tf.float32, shape=(None, DIMENSION))
    #test_labels_placeholder = tf.placeholder(tf.int32, shape=(None))
    feed_dict_test = {data_placeholder: test_data, labels_placeholder: test_labels}
    feed_dict = {data_placeholder: data, labels_placeholder: labels}

    # Train data
    true_count = sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / (N*NUM_CLASSES)
    # Test data
    true_count_test = sess.run(eval_correct, feed_dict=feed_dict_test)
    precision_test = true_count_test / (N*NUM_CLASSES)
    print('Training data   Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (N*NUM_CLASSES, true_count, precision))
    print('Test data   Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (N*NUM_CLASSES, true_count_test, precision_test))


import os
import copy
import numpy as np
import tensorflow as tf

from tools import utils
from models.slim import model
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

# Define os and dataset flags
tf.app.flags.DEFINE_string('data_dir', '/home/christian/Data/ILSVRC/tfrecords/', 'Path to data directory')
tf.app.flags.DEFINE_string('train_dir', '/home/christian/Data/ILSVRC/tfrecords/train-*', 'Path to training data')
tf.app.flags.DEFINE_string('val_dir', '/home/christian/Data/ILSVRC/tfrecords/validation-*', 'Path to validation data')
tf.app.flags.DEFINE_string('log_dir', '/home/christian/YOLO/logs', 'Path to the log folder')
tf.app.flags.DEFINE_integer('train_samples', 1281167, 'Number of training samples in ImageNet')
tf.app.flags.DEFINE_integer('validation_samples', 50000, 'Number of validation samples in ImageNet')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Number of classes in ImageNet, plus one for Null class')

# Define training flags
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, 'Initial learning rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0, 'Number of epochs per decay')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16, 'Learning rate decay factor')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 299, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_epochs', 300, 'Total number of epochs')

# Define learning rate schedule
RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

# Define a list of data files
TRAIN_SHARDS = tf.gfile.Glob(FLAGS.train_dir)
VAL_SHARDS = tf.gfile.Glob(FLAGS.val_dir)

def train():
  with tf.Graph().as_default():
    images, labels = utils.load_batch(batch_size=FLAGS.batch_size, 
                                      num_epochs=FLAGS.num_epochs, 
                                      shards=TRAIN_SHARDS)

    # Define model
    predictions = model.tiny_yolo(images, pretrain=True)

    # Define loss function and optimizer
    loss = slim.losses.softmax_cross_entropy(predictions, labels)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

    # Create training op
    train_op = slim.learning.create_train_op(loss, optimizer)

    # Initialize training
    slim.learning.train(train_op, 
                        FLAGS.log_dir,
                        number_of_steps=FLAGS.max_steps,
                        save_summaries_secs=300,
                        save_interval_secs=300)

train()
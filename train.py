# Winning combo:
# 

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
tf.app.flags.DEFINE_string('train_dir', '/home/christian/TinyImagenetYOLO/YOLO/datasets/tiny-imagenet-200/cached/train.tfrecords', 'Path to training data')
tf.app.flags.DEFINE_string('val_dir', '/home/christian/TinyImagenetYOLO/YOLO/datasets/tiny-imagenet-200/cached/test.tfrecords', 'Path to validation data')
tf.app.flags.DEFINE_string('log_dir', '/home/christian/TinyImagenetYOLO/YOLO/logs', 'Path to the log folder')
tf.app.flags.DEFINE_string('trainlog_dir', '/home/christian/TinyImagenetYOLO/YOLO/logs/train', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_dir', '/home/christian/TinyImagenetYOLO/YOLO/logs/eval', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_integer('train_samples', 1281167, 'Number of training samples in ImageNet')
tf.app.flags.DEFINE_integer('validation_samples', 50000, 'Number of validation samples in ImageNet')
tf.app.flags.DEFINE_integer('num_classes', 200, 'Number of classes in Tiny ImageNet')

# Define training flags
tf.app.flags.DEFINE_float('initial_learning_rate', 0.00001, 'Initial learning rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum optimizer')
tf.app.flags.DEFINE_float('adam_epsilon', 0.1, 'Stability value for adam optimizer')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', None, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Total number of epochs')

# Define a list of data files
TRAIN_SHARDS = FLAGS.train_dir
VAL_SHARDS = FLAGS.val_dir

# SMALLNET MODELS ################################################################################
# Smallnet 1: LR = 0.100,      GD,     L2 = 0.005,  ReLU, Stable but slow, 94k iters, 10% accuracy
# Smallnet 2: LR = 0.010,      GD,     L2 = 0.005,  ReLU, Stable but slow, 30k iters,  5% accuracy
# Smallnet 3: LR = 0.010,      GD,     L2 = 0.005,  ELU,  ?
# Smallnet 4: LR = 0.010,      Adam,   L2 = 0.005,  ReLU, epsilon = 0.1, ?
# Smallnet 5: LR = 0.010,      Adagrad,L2 = 0.005,  ReLU, ?
##################################################################################################

#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction=0.5 # don't hog all vRAM

# Momentum optimizer with log loss (using nesterov and a lower initial momentum)
# Replace nesterov=False and momentum=0.9 for the best momentum classifier so far
def train_momentum_cross_entropy():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True)

        labels = tf.one_hot(labels, depth=200)
        print_op = tf.Print(input_=labels,
                            data=[labels])

        # Define model
        #predictions = model.tiny_yolo(images, is_training=True, pretrain=True)
        predictions = model.AlexNet(images, is_training=True)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                               momentum=FLAGS.momentum,
                                               use_nesterov=True)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30)

# Gradient descent optimizer with log loss
# LR = 0.05 from steps 0 - 22.5k
# LR = 0.02 from steps 22.5k
def train_gd_cross_entropy():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True)

        labels = tf.one_hot(labels, depth=200)

        # Define model
        # predictions = model.tiny_yolo(images, is_training=True, pretrain=True)
        # predictions = model.simplenetC(images, softmax=True)
        predictions = model.smallnet1(images)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.initial_learning_rate)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30)


def train_adadelta_cross_entropy():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True)

        labels = tf.one_hot(labels, depth=200)

        # Define model
        # predictions = model.tiny_yolo(images, is_training=True, pretrain=True)
        # predictions = model.simplenetC(images, softmax=True)
        predictions = model.smallnet(images)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.initial_learning_rate)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30)

# Adam optimizer with log loss
def train_adam_cross_entropy():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True)

        labels = tf.one_hot(labels, depth=200)

        # Define model
        predictions = model.smallnet4(images)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                           epsilon=FLAGS.adam_epsilon)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30)


# RMSProp with Momentum
def train_rmsprop_momentum_cross_entropy():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True)

        labels = tf.one_hot(labels, depth=200)

        # Define model
        predictions = model.tiny_yolo(images, is_training=True, pretrain=True)
        # predictions = model.simplenet(images, softmax=True, is_training=True)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                              epsilon=FLAGS.adam_epsilon,
                                              momentum=FLAGS.momentum)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)
        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30)

train_momentum_cross_entropy()
#train_gd_cross_entropy()
#train_adadelta_cross_entropy()
#train_adam_cross_entropy()
#train_rmsprop_momentum_cross_entropy()
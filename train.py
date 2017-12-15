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
tf.app.flags.DEFINE_string('log_dir', '/home/christian/Data/ILSVRC/logs', 'Path to the log folder')
tf.app.flags.DEFINE_string('trainlog_dir', '/home/christian/Data/ILSVRC/logs/train', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_dir', '/home/christian/Data/ILSVRC/logs/eval', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_integer('train_samples', 1281167, 'Number of training samples in ImageNet')
tf.app.flags.DEFINE_integer('validation_samples', 50000, 'Number of validation samples in ImageNet')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Number of classes in ImageNet')

# Define training flags
tf.app.flags.DEFINE_float('initial_learning_rate', 0.02, 'Initial learning rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum optimizer')
tf.app.flags.DEFINE_float('adam_epsilon', 0.1, 'Stability value for adam optimizer')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', None, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Total number of epochs')

# Define a list of data files
TRAIN_SHARDS = tf.gfile.Glob(FLAGS.train_dir)
VAL_SHARDS = tf.gfile.Glob(FLAGS.val_dir)

#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction=0.5 # don't hog all vRAM

# Momentum optimizer with log loss
def train_momentum_logloss():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(batch_size=FLAGS.batch_size, 
                                          num_epochs=FLAGS.num_epochs, 
                                          shards=TRAIN_SHARDS)

        labels = tf.one_hot(labels, depth=1000)

        # Define model
        # predictions = model.tiny_yolo(images, is_training=True, pretrain=True)
        predictions = model.simplenet(images, softmax=True)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)

        # Define optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                               momentum=FLAGS.momentum)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30)
                            #session_config=config)

# Gradient descent optimizer with log loss
def train_gd_logloss():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(batch_size=FLAGS.batch_size, 
                                          num_epochs=FLAGS.num_epochs, 
                                          shards=TRAIN_SHARDS)

        labels = tf.one_hot(labels, depth=1000)

        # Define model
        # predictions = model.tiny_yolo(images, is_training=True, pretrain=True)
        predictions = model.simplenet(images, softmax=True)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)

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

# Adam optimizer with log loss
def train_adam_logloss():
    with tf.Graph().as_default():
        images, labels = utils.load_batch(batch_size=FLAGS.batch_size, 
                                          num_epochs=FLAGS.num_epochs, 
                                          shards=TRAIN_SHARDS)

        labels = tf.one_hot(labels, depth=1000)

        # Define model
        # predictions = model.tiny_yolo(images, is_training=True, pretrain=True)
        predictions = model.simplenet(images, softmax=True, is_training=True)

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(labels, predictions)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                           epsilon=FLAGS.adam_epsilon)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)
        #train_op = slim.optimize_loss(loss,
        #                              slim.get_global_step(),
        #                              learning_rate=FLAGS.initial_learning_rate,
        #                              optimizer='Adam')

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30)

train_adam_logloss()
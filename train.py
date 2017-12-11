import os
import copy
import numpy as np
import tensorflow as tf

from tools import utils

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

# Define learning rate schedule
RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

# Define a list of data files
TRAIN_SHARDS = tf.gfile.Glob(FLAGS.train_dir)
VAL_SHARDS = tf.gfile.Glob(FLAGS.val_dir)

def train():
  pass
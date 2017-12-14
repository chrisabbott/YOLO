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
metrics = tf.contrib.metrics

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
tf.app.flags.DEFINE_float('initial_learning_rate', 0.005, 'Initial learning rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 224, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', 400, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Total number of epochs')
tf.app.flags.DEFINE_integer('num_evals', 40, 'Number of batches to evaluate')

# Define a list of data files
TRAIN_SHARDS = tf.gfile.Glob(FLAGS.train_dir)
VAL_SHARDS = tf.gfile.Glob(FLAGS.val_dir)

def evaluate():
  with tf.Graph().as_default():
    images, labels = utils.load_batch(batch_size=FLAGS.batch_size, 
                                      num_epochs=FLAGS.num_epochs, 
                                      shards=VAL_SHARDS,
                                      train=False)

    predictions = model.tiny_yolo(images, pretrain=True)
    predictions = tf.to_int32(tf.argmax(predictions, 1))

    metrics_to_values, metrics_to_updates = metrics.aggregate_metric_map({
        'mse': metrics.streaming_mean_squared_error(predictions, labels),
        'accuracy': metrics.streaming_accuracy(predictions, labels),
        })

    for metric_name, metric_value in metrics_to_values.items():
        tf.summary.scalar(metric_name, metric_value)

    slim.evaluation.evaluation_loop(
        '',
        FLAGS.trainlog_dir,
        FLAGS.evallog_dir,
        num_evals=FLAGS.num_evals,
        eval_op = list(metrics_to_updates.values()),
        eval_interval_secs=30)
    
evaluate()
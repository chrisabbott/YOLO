import os
import numpy as np
import skimage.io as io
import tensorflow as tf

from tools import utils
from models.slim import model
import tensorflow.contrib.slim.nets as nets

vgg = nets.vgg
FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

# Define os and dataset flags
tf.app.flags.DEFINE_string('dattrain.pya_dir', os.path.expanduser('~/Data/ILSVRC/tfrecords/'), 'Path to data directory')
tf.app.flags.DEFINE_string('train_dir', os.path.expanduser('~/Data/ILSVRC/tfrecords/train-*'), 'Path to training data')
tf.app.flags.DEFINE_string('val_dir', os.path.expanduser('~/Data/ILSVRC/tfrecords/validation-*'), 'Path to validation data')
tf.app.flags.DEFINE_string('log_dir', os.path.expanduser('~/Data/ILSVRC/logs'), 'Path to the log folder')
tf.app.flags.DEFINE_string('trainlog_dir', os.path.expanduser('~/Data/ILSVRC/logs/train'), 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_dir', os.path.expanduser('~/Data/ILSVRC/logs/eval'), 'Path to the evaluation log folder')
tf.app.flags.DEFINE_integer('train_samples', 1281167, 'Number of training samples in ImageNet')
tf.app.flags.DEFINE_integer('validation_samples', 50000, 'Number of validation samples in ImageNet')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Number of classes in ImageNet')

# Define training flags
tf.app.flags.DEFINE_float('initial_learning_rate', 0.005, 'Initial learning rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 224, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', 400, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Total number of epochs')

# Define a list of data files
TRAIN_SHARDS = tf.gfile.Glob(FLAGS.train_dir)
VAL_SHARDS = tf.gfile.Glob(FLAGS.val_dir)

def main():
    # image = io.imread('/home/basim/shared/comp4107/YOLO/datasets/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG')
    image = io.imread('./datasets/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG')
    io.imshow(image)
    print('ayy')

if __name__ == '__main__':
    main()

import tensorflow as tf
import os

from tensorflow.python.ops import array_ops

slim = tf.contrib.slim

def tiny_yolo(inputs, is_training=True, pretrain=False):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    
    net = slim.conv2d(inputs, 16, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 32, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.batch_norm(net, is_training=is_training)
    net = slim.conv2d(net, 256, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.dropout(net, is_training=is_training)
    net = slim.conv2d(net, 512, [3,3])
    net = slim.max_pool2d(net, [2,2], stride=1)
    net = slim.dropout(net, is_training=is_training)
    net = slim.conv2d(net, 1024, [3,3])

    if pretrain:
      net = slim.avg_pool2d(net, [2,2], stride=1)
      net = slim.flatten(net)
      net = slim.fully_connected(net, 1000, activation_fn=slim.softmax)
      return net

    net = slim.conv2d(net, 512, [3,3])
    net = slim.conv2d(net, 425, [3,3])
    net = slim.fully_connected(net, 4096)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 7 * 7 * 30, activation_fn=slim.softmax)
    return net


def simplenet(inputs, softmax=False, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                      weights_regularizer=slim.l2_regularizer(0.005),
                      stride=1):

    net = slim.conv2d(inputs, 64, [3,3])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [1,1])
    net = slim.dropout(net, is_training=is_training)

    net = slim.conv2d(net, 128, [1,1])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.dropout(net, is_training=is_training)

    if softmax:
      net = slim.conv2d(net, 1000, [1,1], activation_fn=slim.softmax)
      net = array_ops.squeeze(net, [1, 2])
      return net
    else:
      net = slim.conv2d(net, 1000, [1,1], activation_fn=None)
      net = array_ops.squeeze(net, [1, 2])
      return net
    

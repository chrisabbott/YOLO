import tensorflow as tf
import os

slim = tf.contrib.slim

#https://github.com/mnuke/tf-slim-mnist/blob/master/

def tiny_yolo(inputs, pretrain=False):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    
    net = slim.conv2d(inputs, 16, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 32, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 256, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 512, [3,3])
    net = slim.max_pool2d(net, [2,2], stride=1)
    net = slim.conv2d(net, 1024, [3,3])

    if pretrain:
      net = slim.avg_pool2d(net, [2,2], stride=1)
      net = slim.fully_connected(net, 1000)
      return net

    net = slim.conv2d(net, 512, [3,3])
    net = slim.conv2d(net, 425, [3,3])
    net = slim.fully_connected(net, 4096)
    net = slim.fully_connected(net, 7 * 7 * 30)
    return net
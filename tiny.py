import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

batch_size = 256
test_size = 256
epochs = 15
logs_path = './artifacts/'


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def _conv2d(x, weights, name, strides=[1, 1, 1, 1], padding='SAME'):
    with tf.name_scope(name):
        return tf.nn.conv2d(x,
                            weights,
                            strides=strides,
                            padding=padding,
                            name=name)


def _relu(x, name):
    with tf.name_scope(name):
        return tf.nn.relu(x)


def _max_pool(x, name, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x,
                              ksize=ksize,
                              strides=strides,
                              padding=padding,
                              name=name)


def _dropout(x, sigma, name):
    with tf.name_scope(name):
        return tf.nn.dropout(x, sigma)


def _matmul(x, y, name):
    with tf.name_scope(name):
        return tf.matmul(x, y)


def model(x, w_1, w_2, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = _conv2d(x, w_1, name='conv1')
    l1 = _relu(l1a, name='relu1')
    l1 = _max_pool(l1, name='pool1')
    l1 = _dropout(l1, sigma=p_keep_conv, name='dropout1')

    l2 = _conv2d(l1, w_2, name='conv2')
    l2 = _relu(l2, name='relu2')
    l2 = _max_pool(l2, name='pool2')
    l2 = _dropout(l2, sigma=p_keep_conv, name='dropout2')

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])
    l3 = _dropout(l3, sigma=p_keep_conv, name='dropout3')

    l4 = _matmul(l3, w_fc, name='matmul1')
    l4 = _relu(l4, name='relu3')
    l4 = _dropout(l4, sigma=p_keep_hidden, name='dropout4')

    pyx = _matmul(l4, w_o, name='pyx')
    return pyx


def main():
    dataset_dir = os.path.join(os.getcwd(), 'datasets/tiny-imagenet-200/cached/')
    train_x_file = os.path.join(dataset_dir, 'train_images.npy')
    train_y_file = os.path.join(dataset_dir, 'train_labels.npy')
    test_x_file = os.path.join(dataset_dir, 'test_images.npy')
    test_y_file = os.path.join(dataset_dir, 'test_labels.npy')

    train_x = np.load(train_x_file)
    test_x = np.load(test_x_file)

    train_y = np.load(train_y_file)
    train_y = np.resize(train_y, (train_y.shape[0], 1))
    test_y = np.load(test_y_file)
    test_y = np.resize(test_y, (test_y.shape[0], 1))

    encoder = OneHotEncoder()
    train_y = encoder.fit(train_y).transform(train_y).toarray()
    test_y = encoder.fit(test_y).transform(test_y).toarray()

    x = tf.placeholder("float", [None, 64, 64, 3])
    y = tf.placeholder("float", [None, 200])

    w_1 = init_weights([5, 5, 3, 32])  # 3x3x3 conv, 32 outputs
    w_2 = init_weights([5, 5, 32, 64])
    w_fc = init_weights([64 * 8 * 8, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    w_o = init_weights([625, 200])  # FC 625 inputs, 10 outputs (labels)

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(x, w_1, w_2, w_fc, w_o, p_keep_conv, p_keep_hidden)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y))

    with tf.name_scope("train_op"):
        train_op = tf.train.RMSPropOptimizer(0.02, 0.9).minimize(cost)

    with tf.name_scope("predict_op"):
        predict_op = tf.argmax(py_x, 1)
        tf.summary.scalar("predict_op", predict_op)

    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs_path, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(epochs):
            batch_starts = range(0, len(train_x), batch_size)
            batch_ends = range(batch_size, len(train_x) + 1, batch_size)
            training_batches = zip(batch_starts, batch_ends)

            for start, end in training_batches:
                train_graph = sess.run(train_op, feed_dict={
                    x: train_x[start:end],
                    y: train_y[start:end],
                    p_keep_conv: 0.6,
                    p_keep_hidden: 0.9,
                })

            test_indices = np.arange(len(test_x))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            labels = np.argmax(test_y[test_indices], axis=1)
            predictions = sess.run(predict_op, feed_dict={
                x: test_x[test_indices],
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0
            })

            accuracy = np.mean(labels == predictions)
            print("Epoch: {}  Accuracy: {}".format(i + 1, accuracy))


if __name__ == '__main__':
    main()

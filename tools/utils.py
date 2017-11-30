import os
#import cv2
import numpy as np

'''
def _load_X():
    label_dir = 'VOCdevkit/VOC2012/labels/'
    image_dir = 'VOCdevkit/VOC2012/JPEGImages/'
    X = []

    if not os.path.exists(label_dir):
        print("Generate labels before loading dataset.")
        return

    if not os.path.exists(image_dir):
        print("No image directory found.")
        return

    for i in os.listdir(label_dir):
        path = "%s%s.jpg" % (image_dir,
                             os.path.splitext(i)[0])
        X.append(cv2.imread(path, 1))

    return X
'''

def _format_y():
    pass

def _load_y():
    pass

def load_pascal():
    X = _load_X()
    pass

def read_images_from_lmdb(db_name, visualize):
    env = lmdb.open(db_name)
    txn = env.begin()
    cursor = txn.cursor()
    X = []
    y = []
    idxs = []
    for idx, (key, value) in enumerate(cursor):
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        X.append(np.array(datum_to_array(datum)))
        y.append(datum.label)
        idxs.append(idx)
    if visualize:
        print("Visualizing a few images...")
        for i in range(9):
            img = X[i]
            plt.subplot(3,3,i+1)
            plt.imshow(img)
            plt.title(y[i])
            plt.axis('off')
        plt.show()
    return X, y, idxs

def imagenet_pipeline(env, keys, epochs=90, batch_size=256):
    # Create a queue of keys
    producer = tf.train.string_input_producer(keys, num_epochs=epochs)
    key = producer.dequeue()

    def retrieve_batch(key):
        with env.begin() as txn:
            val = txn.get(key)

        X_s = get_example_from_val(val)
        y_s = get_label_from_val(val)

        return X_s, y_s

    # Single X and y value pair
    X_s, y_s = tf.py_func(retrieve_batch, [key], [tf.float64, tf.float64])
    X, y = tf.train.batch([X_s, y_s], batch_size)

    return X, y

def main():
    load_pascal()

if __name__ == '__main__':
    main()
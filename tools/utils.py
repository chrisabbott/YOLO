import os
import lmdb
import numpy as np
import tensorflow as tf
from tools.protobuf import datum_pb2

PATH = "/home/christian/Data/ILSVRC15/processed/ilsvrc12_val_lmdb/"

''' Generate a list of all keys in lmdb environment
        Args: lmdb
        Returns: List of keys in lmdb environment
'''
def get_keys(env, n=0):
  keys = []
  
  with env.begin() as txn:
    cursor = txn.cursor()

    i = 0

    for key, value in cursor:
      if i >= n and n > 0:
        break;

      keys.append(key)

  return keys


''' Decode an entry in lmdb environment by key
        Args: lmdb, key to decode
        Returns: decoded entry in lmdb
'''
def _decode(env, key):
  image = False
  label = False

  with env.begin() as txn:
    raw = txn.get(key)
    datum = datum_pb2.Datum()
    datum.ParseFromString(raw)

    label = datum.label

    if datum.data:
      image = np.fromstring(datum.data, dtype=np.uint8).reshape(datum.channels, datum.height, datum.width).transpose(1,2,0)
    else:
      image = np.array(datum.float_data).astype(np.float).reshape(datum.channels, datum.height, datum.width).transpose(1,2,0)

  return image, label, key


''' Specify a training data pipeline that randomly generates batches of tensors
        Args: lmdb, list of keys, epochs, batch size
        Returns: List of tensors
'''
def data_pipeline(env, keys, epochs=90, batch_size=256):
    # Create a queue of keys
  producer = tf.train.string_input_producer(keys, num_epochs=epochs, shuffle=True)
  key = producer.dequeue()

  def retrieve_batch(key):
      with env.begin() as txn:
        X_s, y_s, key = _decode(env, key)

      return X_s, y_s

  # Single X and y value pair
  X_s, y_s = tf.py_func(retrieve_batch, [key], [tf.float64, tf.float64])
  X_s.set_shape([224,224,3])
  y_s.set_shape([1])
  
  X, y = tf.train.batch([X_s, y_s], batch_size)

  return X, y
import lmdb
import numpy as np
import datum_pb2

path = ""

def get_keys(n=0):
  env = lmdb.open(path, readonly=True)
  keys = []

  with env.begin() as txn:
    cursor = txn.cursor()

    i = 0

    for key, value in cursor:
      if i >= n and n > 0:
        break;

      keys.append(key)

  return keys

def read_single(key):
  image = False
  label = False

  env = lmdb.open(path, readonly=True)
  
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
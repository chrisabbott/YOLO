import lmdb

def main():
  env = lmdb.open("/home/christian/Data/ILSVRC15/processed/ilsvrc12_val_lmdb")
  txn = env.begin()
  cursor = txn.cursor()

  print(txn.stat()['entries'])

if __name__ == '__main__':
    main()
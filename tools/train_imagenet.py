import tensorflow as tf
import lmdb
import utils

def main():
	# Initialize variables
	init_op = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init_op)

	# Initialize queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# Initialize training data path and lmdb environment
	# train_path = "/home/christian/Data/ILSVRC15/processed/ilsvrc12_train_lmdb"
	# train_env = lmdb.open(train_path, readonly=True)
	# train_keys = utils.get_keys(train_env, n=1)

	# Initialize validation data path and lmdb environment
	val_path = "/home/christian/Data/ILSVRC15/processed/ilsvrc12_val_lmdb"
	val_env = lmdb.open(val_path, readonly=True)
	val_keys = utils.get_keys(val_env, n=1)

	# Initialize data pipe
	X,y = utils.data_pipeline(val_env, val_keys, epochs=90, batch_size=256)

	try:
		while not coord.should_stop():
			# Train here
			pass

	except tf.error.OutOfRangeError:
		print("Training completed.")

	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()
	

if __name__ == '__main__':
    main()
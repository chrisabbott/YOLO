import tensorflow as tf
import utils

def main():
	# Initialize variables
	init_op = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init_op)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	file = "~/Data/ILSVRC15/processed/ilsvrc12_val_lmdb"
	pipe = utils.imagenet_pipeline()

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
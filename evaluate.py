from tools import utils
from models.slim import model
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
metrics = tf.contrib.metrics

# Define os and dataset flags
tf.app.flags.DEFINE_string('data_dir', '/home/christian/Data/ILSVRC/tfrecords/', 'Path to data directory')
tf.app.flags.DEFINE_string('train_dir', '/home/christian/TinyImagenetYOLO/YOLO/datasets/tiny-imagenet-200/cached/train.tfrecords', 'Path to training data')
tf.app.flags.DEFINE_string('val_dir', '/home/christian/TinyImagenetYOLO/YOLO/datasets/tiny-imagenet-200/cached/test.tfrecords', 'Path to validation data')
tf.app.flags.DEFINE_string('log_dir', '/home/christian/TinyImagenetYOLO/YOLO/logs', 'Path to the log folder')
tf.app.flags.DEFINE_string('trainlog_dir', '/home/christian/TinyImagenetYOLO/YOLO/logs/train', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_dir', '/home/christian/TinyImagenetYOLO/YOLO/logs/eval', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_integer('num_classes', 200, 'Number of classes in Tiny ImageNet')

# Define training flags
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, 'Initial learning rate')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', 400, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Total number of epochs')
tf.app.flags.DEFINE_integer('num_evals', 20, 'Number of batches to evaluate')

# Define a list of data files
TRAIN_SHARDS = FLAGS.train_dir
VAL_SHARDS = FLAGS.val_dir

def evaluate():
  with tf.Graph().as_default():
    config = tf.ConfigProto(device_count={'GPU':0})

    images, labels = utils.load_batch(shards=VAL_SHARDS,
                                      batch_size=FLAGS.batch_size,
                                      train=False)

    print_op = tf.Print(input_=labels,
                        data=[labels])

    predictions = model.AlexNetXL(images, is_training=False)
    predictions = tf.to_int64(tf.argmax(predictions, 1))

    metrics_to_values, metrics_to_updates = metrics.aggregate_metric_map({
        'mse': metrics.streaming_mean_squared_error(predictions, labels),
        'rmse': metrics.streaming_root_mean_squared_error(predictions, labels),
        'accuracy': metrics.streaming_accuracy(predictions, labels),
        'precision': metrics.streaming_precision(predictions, labels),
    })

    for metric_name, metric_value in metrics_to_values.items():
        tf.summary.scalar(metric_name, metric_value)

    slim.evaluation.evaluation_loop(
        '',
        FLAGS.trainlog_dir,
        FLAGS.evallog_dir,
        num_evals=FLAGS.num_evals,
        eval_op = list(metrics_to_updates.values()),
        eval_interval_secs=5,
        session_config=config)

evaluate()

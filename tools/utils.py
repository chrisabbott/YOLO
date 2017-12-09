import tensorflow as tf
import os


_FILE_PATTERN = 'train-%s-of-%s.tfrecord'
_SPLITS_TO_SIZES = {'train': 1281167, 'test': 50000}
_NUM_CLASSES = 1000
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [224 x 224 x 3] RGB image.',
    'label': 'A single integer between 0 and 999',
}

def preprocess(image, output_height, output_width, is_training):
  image = tf.to_float(image)
  image = tf.image.resize_image_with_crop_or_pad(
    image, output_width, output_height)
  image = tf.subtract(image, 128.0)
  image = tf.div(image, 128.0)
  return image

def load_batch(dataset, batch_size=32, height=224, width=224, is_training=False):
  data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

  image, label = data_provider.get(['image', 'label'])
  image = preprocess(image, height, width, is_training)

  images, labels = tf.train.batch(
    [image, label],
    batch_size=batch_size,
    allow_smaller_final_batch=True)

  return images, labels
import tensorflow as tf
from tensorflow import keras

# load dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# normalise images
train_images = train_images / 255.0
test_images = test_images / 255.0

# create a dataset (iterable) from the data using a specified batch size
batch_size = 128
dataset = tf.data.Dataset.from_tensor_slices(train_images)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
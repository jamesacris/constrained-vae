import tensorflow_datasets as tfds

dataset, ds_info = tfds.load("Dsprites", split="train", with_info=True)

# Show examples
# fig = tfds.show_examples(dataset, ds_info)

# create a dataset (iterable) from the data using a specified batch size
# NB: dataset is already normalised as the images are binary
batch_size = 128
dataset = dataset.shuffle(buffer_size=737280).batch(batch_size)
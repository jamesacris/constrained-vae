import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import matplotlib.pyplot as plt

# Configure gpu
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

from helpers import subplot_image
from beta_vae_model import latent_dim
from data_prep import dataset, batch_size

# load encoder & decoder
encoder = tf.keras.models.load_model("./models/BVAE_encoder_cnn_20epochs_4.0beta")
encoder.summary()

decoder = tf.keras.models.load_model("./models/BVAE_decoder_cnn_20epochs_4.0beta")
decoder.summary()


# Visualise some images at random

# seed
nrows = 4
ncols = 8
# seed = tf.random.normal([nrows * ncols, latent_dim], mean=0, stddev=1)

# decoded images
# decoded_images = decoder.predict(seed)

# reconstruct some images from the dataset
np_ds = tfds.as_numpy(dataset)
for step, ex in enumerate(dataset):
    img = ex["image"]
    if step >= (nrows * ncols):
        break

print(img.shape)
print(type(img))

# show imgs
# plt.figure(dpi=100, figsize=(ncols * 2, nrows * 2.2))
# for iplot in range(nrows * ncols):
#     subplot_image(np_im[iplot, :, :], "", nrows, ncols, iplot)
# plt.savefig("./figs/samples.png")
# plt.show()

# encode & decode
encoded_imgs = encoder.predict(img[0 : nrows * ncols, :, :])
decoded_imgs = decoder.predict(
    encoded_imgs[2]
)  # index last of (z_mean, z_var, z_output)

# plot images
plt.figure(dpi=100, figsize=(ncols * 2, nrows * 2.2))
for iplot in range(nrows * ncols):
    subplot_image(decoded_imgs[iplot, :, :], "", nrows, ncols, iplot)
# plt.savefig("./figs/BVAE/reconstructions_dense_cnn_20epochs_4beta.png")
plt.show()


# evaluate disentanglement

labels = [
    'label_orientation',
    'label_scale',
    'label_shape',
    'label_x_position',
    'label_y_position'
]

# no. of batches to avaluate z_diff for
batches = 2

# choose a generative factor
i = np.random.randint(5)
label = labels[i]
print(label)

z_diffs = []
enum_data = enumerate(dataset)
for step, batch in enum_data:
    # iterate over (batches) batch
    if step >= batches:
        break
    value_batch = batch[label].numpy()
    image_batch = batch['image'].numpy()

    z_diffs_batch = []
    for j in range(batch_size):
        if j>=2:
            break
        _value_batch = np.delete(value_batch, j)
        index = np.where(_value_batch==value_batch[j])[0][0]
        if index>=j:
            index+=1

        img_pair = tf.reshape(tf.constant([image_batch[j], image_batch[index]]), [2, 64, 64])

        # encode
        z = encoder.predict(img_pair)
        z_mean = z[0]

        # compute z_diff
        z_diff = abs(z_mean[0] - z_mean[1])

        z_diffs_batch.append(z_diff)

    z_diffs_batch = np.mean(np.array(z_diffs_batch), axis=0)
    z_diffs.append(z_diffs_batch)

print(z_diffs)
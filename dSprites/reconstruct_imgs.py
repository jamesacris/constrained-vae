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

nrows = 4
ncols = 8

# reconstruct some images from the dataset
np_ds = tfds.as_numpy(dataset)
for step, ex in enumerate(dataset):
    img = ex["image"]
    if step >= (nrows * ncols):
        break

# encode & decode
encoded_imgs = encoder.predict(img[0 : nrows * ncols, :, :])
decoded_imgs = decoder.predict(encoded_imgs[2])

# plot images
plt.figure(dpi=100, figsize=(ncols * 2, nrows * 2.2))
for iplot in range(nrows * ncols):
    subplot_image(decoded_imgs[iplot, :, :], "", nrows, ncols, iplot)
# plt.savefig("./figs/BVAE/reconstructions_dense_cnn_20epochs_4beta.png")
plt.show()
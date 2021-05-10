import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt

from helpers import subplot_image
from beta_vae_model import latent_dim
from data_prep import dataset

# load encoder & decoder
encoder = tf.keras.models.load_model("./models/BVAE_encoder_dense_50ep")
encoder.summary()

decoder = tf.keras.models.load_model("./models/BVAE_decoder_dense_50ep")
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
plt.savefig("./figs/BVAE/reconstructions_dense_50epoch.png")
plt.show()
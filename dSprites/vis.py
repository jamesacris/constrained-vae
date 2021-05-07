import tensorflow as tf
import os
import matplotlib.pyplot as plt

from helpers import subplot_image
from beta_vae_model import latent_dim

# load encoder & decoder
encoder = tf.keras.models.load_model('./models/BVAE_encoder_initial')
encoder.summary()

decoder = tf.keras.models.load_model('./models/BVAE_decoder_initial')
decoder.summary()


# Visualise some images at random

# seed
nrows = 4
ncols = 8
seed = tf.random.normal([nrows * ncols, latent_dim], mean=0, stddev=20)

# decoded images
decoded_images = decoder.predict(seed)

# plot images
plt.figure(dpi=100, figsize=(ncols * 2, nrows * 2.2))
for iplot in range(nrows * ncols):
    subplot_image(decoded_images[iplot, :, :], '', nrows, ncols, iplot)
plt.show()
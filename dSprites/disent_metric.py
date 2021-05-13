import tensorflow as tf
import tensorflow_datasets as tfds
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

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
from data_prep import dataset

# load encoder & decoder
encoder = tf.keras.models.load_model("./models/BVAE_encoder_cnn_20epochs_4.0beta")
encoder.summary()

decoder = tf.keras.models.load_model("./models/BVAE_decoder_cnn_20epochs_4.0beta")
decoder.summary()

# evaluate disentanglement

# Change figure aesthetics
sns.set_context("talk", font_scale=1.2, rc={"lines.linewidth": 1.5})

# Load dataset
dataset_zip = np.load(
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    allow_pickle=True,
    encoding="latin1",
)

print("Keys in the dataset:", dataset_zip.keys())
imgs = dataset_zip["imgs"]
latents_values = dataset_zip["latents_values"]
latents_classes = dataset_zip["latents_classes"]
metadata = dataset_zip["metadata"][()]

print("Metadata: \n", metadata)

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata["latents_sizes"]
latents_names = metadata["latents_names"]
latents_bases = np.concatenate(
    (
        latents_sizes[::-1].cumprod()[::-1][1:],
        np.array(
            [
                1,
            ]
        ),
    )
)


def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)
    return samples


# Helper function to show images
def show_images_grid(imgs_, num_images=25):
    ncols = int(np.ceil(num_images ** 0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()
    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap="Greys_r", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")


def show_density(imgs):
    _, ax = plt.subplots()
    ax.imshow(imgs.mean(axis=0), interpolation="nearest", cmap="Greys_r")
    ax.grid("off")
    ax.set_xticks([])
    ax.set_yticks([])


# Conditional sampling

latent_label_to_fix = "posX"
_i = latents_names.index(latent_label_to_fix)

batches = 2
batch_size = 128

z_diffs = []
for i in range(batches):
    z_diffs_batch = []
    for j in range(batch_size):

        latent_value_index_fixed = np.random.randint(0, latents_sizes[_i])

        latents_sampled = sample_latent(size=2)
        latents_sampled[:, _i] = latent_value_index_fixed
        indices_sampled = latent_to_index(latents_sampled)
        imgs_sampled = imgs[indices_sampled]

        img_pair = tf.convert_to_tensor(imgs_sampled)

        # encode
        z = encoder.predict(img_pair)
        z_mean = z[0]
        # compute z_diff
        z_diff = abs(z_mean[0] - z_mean[1])

        z_diffs_batch.append(z_diff)
    z_diffs_batch = np.mean(np.array(z_diffs_batch), axis=0)
    z_diffs.append(z_diffs_batch)

print(z_diffs)
import tensorflow as tf
import tensorflow_datasets as tfds
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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


def get_zdiffs(batches, batch_size):
    z_diffs = []
    latent_indices = []
    for n, latent_label_to_fix in enumerate(latents_names[1:]):
        latent_index = n + 1

        for i in range(batches):
            z_diffs_batch = []
            for j in range(batch_size):

                # sample the latent variables
                latents_sampled = sample_latent(size=2)

                # fix one of the latent variables (make second equal first)
                latents_sampled[1, latent_index] = latents_sampled[0, latent_index]

                # find the corresponding images
                indices_sampled = latent_to_index(latents_sampled)
                imgs_sampled = imgs[indices_sampled]
                img_pair = tf.convert_to_tensor(imgs_sampled)

                # encode
                z = encoder.predict(img_pair)
                z_mean = z[0]

                # compute z_diff
                z_diff = abs(z_mean[0] - z_mean[1])

                # record
                z_diffs_batch.append(z_diff)

            # compute mean of zdiffs over one batch
            z_diffs_batch = np.mean(np.array(z_diffs_batch), axis=0)

            # record mean zdiff and the corresponding label
            z_diffs.append(z_diffs_batch)
            latent_indices.append(latent_index)
    return {
        "z_diffs": z_diffs,
        "latent_indices": latent_indices,
    }


batch_size = 128

# prep training data
training_data = get_zdiffs(10, batch_size)

x_train = np.array(training_data["z_diffs"])
y_train = np.array(training_data["latent_indices"])

# sklearn linear classifier
classifier = make_pipeline(StandardScaler(), SGDClassifier(loss="log", max_iter=100))

# train
classifier.fit(x_train, y_train)

# get testing data to evaluate disentanglement metric
batches = 1  # 1000 per factor, 5000 total
test_data = get_zdiffs(batches, batch_size)

x_test = np.array(test_data["z_diffs"])
y_test = np.array(test_data["latent_indices"])

# evaluate disentanglement metric
disentanglement_score = classifier.score(x_test, y_test)
print(disentanglement_score)
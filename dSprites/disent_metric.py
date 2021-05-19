from random import shuffle
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


def get_zdiffs(batches, batch_size):
    n_latent_real = len(latents_names[1:])
    z_diffs = np.zeros((n_latent_real, batches, latent_dim))
    latent_indices = np.zeros((n_latent_real, batches), dtype=int)

    for n, latent_label_to_fix in enumerate(latents_names[1:]):
        latent_index = n + 1
        print(n)

        latents_sampled_1 = sample_latent(size=batch_size * batches)
        latents_sampled_2 = sample_latent(size=batch_size * batches)
        print(latents_sampled_1.shape)

        latents_sampled_1[:, latent_index] = latents_sampled_2[:, latent_index]

        indices_sampled_1 = latent_to_index(latents_sampled_1)
        indices_sampled_2 = latent_to_index(latents_sampled_2)
        print(indices_sampled_1.shape)

        imgs_sampled_1 = imgs[indices_sampled_1]
        imgs_sampled_2 = imgs[indices_sampled_2]
        print(imgs_sampled_1.shape)

        # img_pair_1 = tf.convert_to_tensor(imgs_sampled_1)
        # img_pair_2 = tf.convert_to_tensor(imgs_sampled_2)

        z_1 = encoder.predict(imgs_sampled_1)[0]
        z_2 = encoder.predict(imgs_sampled_2)[0]
        print(z_1.shape)

        z_diff = np.abs(z_1 - z_2)
        print(z_diff.shape)

        z_diffs[n, :, :] = np.mean(
            z_diff.reshape((batches, batch_size, latent_dim)), axis=1
        )

        latent_indices[n, :] = n

    shuffle_index = np.arange(0, n_latent_real * batches)
    np.random.shuffle(shuffle_index)
    print(shuffle_index)

    z_diffs = z_diffs.reshape((n_latent_real * batches, latent_dim))[shuffle_index]

    latent_indices = latent_indices.reshape((n_latent_real * batches))[shuffle_index]
    # latent_indices = np.eye(n_latent_real)[latent_indices]
    print(latent_indices)

    return {
        "z_diffs": z_diffs,
        "latent_indices": latent_indices,
    }


batch_size = 64

# prep training data
training_data = get_zdiffs(500, batch_size)
print("got training data for classifier")

x_train = np.array(training_data["z_diffs"])
y_train = np.array(training_data["latent_indices"])

# sklearn linear classifier
classifier = make_pipeline(
    StandardScaler(), SGDClassifier(loss="log", early_stopping=True)
)

# train
classifier.fit(x_train, y_train)

# get testing data to evaluate disentanglement metric
batches = 100  # 1000 per factor, 5000 total
test_data = get_zdiffs(batches, batch_size)

x_test = np.array(test_data["z_diffs"])
y_test = np.array(test_data["latent_indices"])

# evaluate disentanglement metric
disentanglement_score = classifier.score(x_test, y_test)
print(disentanglement_score)
# tensorflow
import tensorflow as tf
from tensorflow.python.keras.engine import training

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

# keras
from tensorflow import keras

# helpers
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("ggplot")

# need certainty to explain some of the results
import random as python_random

python_random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# data preprocessing
from data_prep import dataset, batch_size

# BVAE model
from beta_vae_model import encoder_VAE, decoder_VAE, BVAE

# training loop
from train_steps_BVAE import train_model

from helpers import plot_losses, plot_reconstruction_losses, plot_kld_lossses


###################
# Hyperparameters

beta = 4.0

epochs = 50

###################

# record training history in these lists
training_logs = {
"losses" : [],
"reconstruction_losses" : [],
"kld_losses" : [],
"epoch_times" : []
}

# build the BVAE
vae_model = BVAE(encoder_VAE, decoder_VAE, beta=beta)

# compile the VAE
vae_model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=1e-2))


# TRAIN
train_model(
    vae_model,
    dataset,
    batch_size,
    epochs,
    training_logs
)


# save model
encoder_VAE.save('./models/BVAE_encoder_dense_50ep')
decoder_VAE.save('./models/BVAE_decoder_dense_50ep')

# plot losses
plot_losses(training_logs["losses"])
plot_reconstruction_losses(training_logs["reconstruction_losses"])
plot_kld_lossses(training_logs["kld_losses"])
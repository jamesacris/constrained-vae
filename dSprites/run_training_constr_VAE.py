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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)],
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
from constr_vae_model import encoder_VAE, decoder_VAE, constr_VAE

# training loop
from train_steps_BVAE import train_model

from helpers import plot_losses, plot_reconstruction_losses, plot_kld_lossses


###################
# Hyperparameters

epochs = 1

warmup_iters = 100
l = 1
d = 1
nd = 2

Lambda = tf.Variable(0.0)
learning_rate_lambda = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.01, decay_steps=1, decay_rate=1e-3
)
opt_lambda = tf.keras.optimizers.SGD(learning_rate=learning_rate_lambda)

constrained_variable = 'kld'
constraint_aim = 1.0

hyperparams = f"{epochs}epochs_params{warmup_iters,l,d,nd,constrained_variable,constraint_aim}"

###################

# record training history in these lists
training_logs = {
"losses" : [],
"reconstruction_losses" : [],
"kld_losses" : [],
"epoch_times" : []
}

# build the BVAE
vae_model = constr_VAE(encoder_VAE, decoder_VAE)

# compile the VAE
vae_model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=1e-2))


# TRAIN
train_model(
    vae_model,
    dataset,
    batch_size,
    warmup_iters,
    l,
    d,
    nd,
    epochs,
    Lambda,
    opt_lambda,
    training_logs,
    constraint_aim,
    constrained_variable)


# save model
encoder_VAE.save(f'./models/BVAE_encoder_cnn_{hyperparams}')
decoder_VAE.save(f'./models/BVAE_decoder_cnn_{hyperparams}')

# plot losses
plot_losses(training_logs["losses"], hyperparams)
plot_reconstruction_losses(training_logs["reconstruction_losses"], hyperparams)
plot_kld_lossses(training_logs["kld_losses"], hyperparams)
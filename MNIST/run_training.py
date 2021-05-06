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
import time
import numpy as np
import matplotlib.pyplot as plt

from helpers import log, report, plot_losses, plot_reconstruction_losses, plot_kld_lossses, plot_kld_diffs, plot_lambdas

plt.style.use("ggplot")

# data preprocessing
from data_prep import dataset, batch_size

# constrained vae model
from constr_vae_model import encoder_VAE, decoder_VAE, constr_VAE

# need certainty to explain some of the results
import random as python_random

python_random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

KLD_aim = 1.0  # SWEEP

# build the BVAE
vae_model = constr_VAE(encoder=encoder_VAE, decoder=decoder_VAE, KLD_aim=KLD_aim)

# compile the VAE with the optimizer for model parameters
vae_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001))

# Lagrangian loss function
@tf.function
def lagrangian(reconstr_loss, kld, constraint_target, Lambda, constrained_variable: str='kld'):
    # constrain kld
    if constrained_variable=='kld':
        # constraint h
        h = tf.nn.relu(kld - constraint_target)
        # Lagrangian
        l = reconstr_loss + Lambda * h
    # constrain reconstruction error
    elif constrained_variable=='reconstr_err':
        # constraint h
        h = tf.nn.relu(reconstr_loss - constraint_target)
        # Lagrangian
        l = kld + Lambda * h
    else:
        raise ValueError(f"constrained_variable must be one of ['kld', 'reconstr_err']")
    return tf.reduce_mean(l)


# Warmup training step (this is just train_w_step with lambda = 0)
@tf.function
def warmup_step(x, constrained_variable: str='kld'):
    if isinstance(x, tuple):
        x = x[0]
    with tf.GradientTape() as tape:
        # encoding
        z_mean, z_log_var, z = vae_model.encoder(x)
        # decoding
        x_prime = vae_model.decoder(z)
        # reconstruction error by binary crossentropy loss
        reconstruction_loss = (
            tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime)) * 28 * 28
        )
        # KL divergence
        kld = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        if constrained_variable=='kld':
            loss = reconstruction_loss  # optimise for reconstruction err only
        elif constrained_variable=='reconstr_err':
            loss = kld  # optimise for kld only
        else:
            raise ValueError(f"constrained_variable must be one of ['kld', 'reconstr_err']")
    # apply gradient
    grads = tape.gradient(loss, vae_model.trainable_weights)
    vae_model.optimizer.apply_gradients(zip(grads, vae_model.trainable_weights))

    # metrics log
    logits = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kld,
        "lambda": 0.0,
        "kld_diff": kld - KLD_aim,
    }
    return logits


# Reconstruction training step (updates model params)
@tf.function
def train_w_step(x, Lambda, constraint_aim, constrained_variable: str='kld'):
    if isinstance(x, tuple):
        x = x[0]
    with tf.GradientTape() as tape:
        # encoding
        z_mean, z_log_var, z = vae_model.encoder(x)
        # decoding
        x_prime = vae_model.decoder(z)
        # reconstruction error by binary crossentropy loss
        reconstruction_loss = (
            tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime)) * 28 * 28
        )
        # KL divergence
        kld = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        # loss = lagrangian
        loss = lagrangian(reconstruction_loss, kld, constraint_aim, Lambda, constrained_variable)
    # apply gradient
    grads = tape.gradient(loss, vae_model.trainable_weights)
    vae_model.optimizer.apply_gradients(zip(grads, vae_model.trainable_weights))

    # metrics log
    logits = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kld,
        "lambda": Lambda,
        "kld_diff": kld - KLD_aim,
    }
    return logits


# Constraint training step (updates lambda). Pass optimizer.
@tf.function
def train_lambda_step(x, opt, Lambda, constraint_aim, constrained_variable: str='kld'):
    if isinstance(x, tuple):
        x = x[0]
    with tf.GradientTape() as tape:
        # encoding
        z_mean, z_log_var, z = vae_model.encoder(x)
        # decoding
        x_prime = vae_model.decoder(z)
        # reconstruction error by binary crossentropy loss
        reconstruction_loss = (
            tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime)) * 28 * 28
        )
        # KL divergence
        kld = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        # loss = - lagrangian (SGA)
        loss = -lagrangian(reconstruction_loss, kld, constraint_aim, Lambda, constrained_variable)
    # calculate and apply gradient
    grad = tape.gradient(target=loss, sources=[Lambda])
    # opt = tf.keras.optimizers.Adam()
    opt.apply_gradients(zip(grad, [Lambda]))

    # metrics log
    logits = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kld,
        "lambda": Lambda,
        "kld_diff": kld - KLD_aim,
    }
    return logits


# training parameters
warmup_iters = 100  # SWEEP
l = 1  # SWEEP
d = 1  # SWEEP
nd = 2
epochs = 5
Lambda = tf.Variable(0.0)
learning_rate_lambda = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.01, decay_steps=1, decay_rate=1e-3
)  # SWEEP (decay rate)
opt_lambda = tf.keras.optimizers.SGD(learning_rate=learning_rate_lambda)

# record training history in these lists
training_logs = {
"losses" : [],
"reconstruction_losses" : [],
"kld_losses" : [],
"Lambdas" : [],
"kld_diff" : [],
"epoch_times" : []
}


# Training loop
def train_model(warmup_iters, l, d, epochs, Lambda, opt_lambda, constraint_aim, constrained_variable: str='kld'):
    # warmup
    t_warm_0 = time.time()
    for step, train_image_batch in enumerate(dataset):
        logits = warmup_step(train_image_batch, constrained_variable)
        # log
        log(logits, training_logs)

        if step >= warmup_iters:
            break
    t_warm_1 = time.time()
    print(f"\nWarmup time: {(t_warm_1 - t_warm_0):.2f}s")

    # report after warmup
    print("\nTraining logs at end of warmup:")
    for metric, value in logits.items():
        print(metric, value.numpy())

    steps_lambda = 0
    steps_w = 0

    # start epochs
    for epoch in range(epochs):
        t_epoch_0 = time.time()
        print(f"\nStart of epoch {epoch + 1}")

        enum_data = enumerate(dataset)
        for step, train_image_batch in enum_data:
            # perform one SGA step for lambda
            logits = train_lambda_step(train_image_batch, opt_lambda, Lambda, constraint_aim, constrained_variable)
            steps_lambda += 1
            # log
            log(logits, training_logs)

            # perform l SGD steps for model params
            for i in range(l):
                try:
                    step, train_image_batch = next(enum_data)
                except StopIteration:
                    break
                logits = train_w_step(train_image_batch, Lambda, constraint_aim, constrained_variable)
                steps_w += 1
                # log
                log(logits, training_logs)
                # report every 200 batches.
                if step % 200 == 0:
                    report(batch_size, step, logits)

            # artificially lower d by only updating every nd-th step
            if step % nd == 0:
                # increment l by d
                l = l + d

            # report every 200 batches.
            if step % 200 == 0:
                report(batch_size, step, logits)
        t_epoch_1 = time.time()
        epoch_time = t_epoch_1 - t_epoch_0
        training_logs["epoch_times"].append(epoch_time)
        print(f"\nEpoch time: {epoch_time:.2f}s")

    mean_epoch_time = np.mean(training_logs["epoch_times"])
    print(f"\nTraining complete! Mean epoch time: {mean_epoch_time:.2f}s")


# TRAIN
train_model(warmup_iters, l, d, epochs, Lambda, opt_lambda, constraint_aim=KLD_aim, constrained_variable='kld')


# plot losses
plot_losses(training_logs["losses"])
plot_reconstruction_losses(training_logs["reconstruction_losses"])
plot_kld_lossses(training_logs["kld_losses"])
plot_kld_diffs(training_logs["kld_diff"])
plot_lambdas(training_logs["Lambdas"])
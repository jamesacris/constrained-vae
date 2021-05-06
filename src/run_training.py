# tensorflow
import tensorflow as tf

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


# Hyperparameters
KLD_aim = 1.0  # SWEEP

# build the BVAE
vae_model = constr_VAE(encoder=encoder_VAE, decoder=decoder_VAE, KLD_aim=KLD_aim)

# compile the VAE with the optimizer for model parameters
vae_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001))

# Lagrangian loss function
@tf.function
def lagrangian(reconstr_loss, kld, Lambda):
    # constraint h
    h = tf.nn.relu(kld - KLD_aim)
    # Lagrangian
    l = reconstr_loss + Lambda * h
    return tf.reduce_mean(l)


# Warmup training step (this is just train_w_step with lambda = 0)
@tf.function
def warmup_step(x):
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
        loss = reconstruction_loss  # optimise for reconstruction loss only
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
def train_w_step(x, Lambda):
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
        loss = lagrangian(reconstruction_loss, kld, Lambda)
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
def train_lambda_step(x, opt, Lambda):
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
        loss = -lagrangian(reconstruction_loss, kld, Lambda)
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
losses = []
reconstruction_losses = []
kld_losses = []
Lambdas = []
kld_diff = []
epoch_times = []

# logging helper function
def log(logits):
    losses.append(logits["loss"])
    reconstruction_losses.append(logits["reconstruction_loss"])
    kld_losses.append(logits["kl_loss"])
    Lambdas.append(logits["lambda"])
    kld_diff.append(logits["kld_diff"])


# reporting helper function
def report(step, logits):
    print(f"\nTraining logs at step {step}:")
    for metric, value in logits.items():
        print(metric, value.numpy())
    print("Seen: %d samples" % ((step + 1) * batch_size))


# Training loop
# TODO: write functions for logging and reporting to clean up code
def train_model(warmup_iters, l, d, epochs, Lambda, opt_lambda):
    # warmup
    t_warm_0 = time.time()
    for step, train_image_batch in enumerate(dataset):
        logits = warmup_step(train_image_batch)
        # log
        log(logits)

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
            logits = train_lambda_step(train_image_batch, opt_lambda, Lambda)
            steps_lambda += 1
            # log
            log(logits)

            # perform l SGD steps for model params
            for i in range(l):
                try:
                    step, train_image_batch = next(enum_data)
                except StopIteration:
                    break
                logits = train_w_step(train_image_batch, Lambda)
                steps_w += 1
                # log
                log(logits)
                # report every 200 batches.
                if step % 200 == 0:
                    report(step, logits)

            if step % nd == 0:
                # increment l by d
                l = l + d

            # report every 200 batches.
            if step % 200 == 0:
                report(step, logits)
        t_epoch_1 = time.time()
        epoch_time = t_epoch_1 - t_epoch_0
        epoch_times.append(epoch_time)
        print(f"\nEpoch time: {epoch_time:.2f}s")

    mean_epoch_time = np.mean(epoch_times)
    print(f"\nTraining complete! Mean epoch time: {mean_epoch_time:.2f}s")


# TRAIN
train_model(warmup_iters, l, d, epochs, Lambda, opt_lambda)

# plot losses
def plot_losses(losses):
    plt.figure(dpi=100)
    plt.plot([abs(loss) for loss in losses])
    plt.xlabel("Training step")
    plt.ylabel("Loss (absolute value)")
    # save
    plt.savefig(f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTloss_{KLD_aim}_KLDaim.png")


# plot reconstruction losses
def plot_reconstruction_losses(reconstruction_losses):
    plt.figure(dpi=100)
    plt.plot(reconstruction_losses)
    plt.xlabel("Training step")
    plt.ylabel("Reconstruction Loss")
    # save
    plt.savefig(
        f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTrecnstr_loss_{KLD_aim}_KLDaim.png"
    )


# plot kld losses
def plot_kld_lossses(kld_losses):
    plt.figure(dpi=100)
    plt.plot(kld_losses)
    plt.xlabel("Training step")
    plt.ylabel("KL Loss")
    plt.yscale("log")
    # save
    plt.savefig(
        f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTkld_loss_{KLD_aim}_KLDaim.png"
    )


# plot kld diffs
def plot_kld_diffs(kld_diff):
    plt.figure(dpi=100)
    plt.plot(kld_diff)
    plt.xlabel("Training step")
    plt.ylabel("KLD - KLD_aim")
    plt.yscale("log")
    # save
    plt.savefig(
        f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTkld_diff_{KLD_aim}_KLDaim.png"
    )


# plot lambdas
def plot_lambdas(Lambdas):
    plt.figure(dpi=100)
    plt.plot(Lambdas)
    plt.xlabel("Training step")
    plt.ylabel("Lambda")
    # save
    plt.savefig(
        f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTlambdas_{KLD_aim}_KLDaim.png"
    )


# plot losses
plot_losses(losses)
plt.show()
plot_reconstruction_losses(reconstruction_losses)
plt.show()
plot_kld_lossses(kld_losses)
plt.show()
plot_kld_diffs(kld_diff)
plt.show()
plot_lambdas(Lambdas)
plt.show()
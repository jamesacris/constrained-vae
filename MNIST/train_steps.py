import tensorflow as tf
from tensorflow import keras

import time
import numpy as np

from helpers import log, report

# Lagrangian loss function
@tf.function
def lagrangian(vae_model, Lambda, reconstr_loss, kld):
    # constrain kld
    if vae_model.constr_variable == "kld":
        # constraint h
        h = tf.nn.relu(kld - vae_model.epsilon)
        # Lagrangian
        l = reconstr_loss + Lambda * h
    # constrain reconstruction error
    elif vae_model.constr_variable == "reconstr_err":
        # constraint h
        h = tf.nn.relu(reconstr_loss - vae_model.epsilon)
        # Lagrangian
        l = kld + Lambda * h
    else:
        raise ValueError(f"constrained_variable must be one of ['kld', 'reconstr_err']")
    return tf.reduce_mean(l)


# Warmup training step (this is just train_w_step with lambda = 0)
@tf.function
def warmup_step(x, vae_model):
    if isinstance(x, tuple):
        x = x[0]
    with tf.GradientTape() as tape:
        # encoding
        z_mean, z_log_var, z = vae_model.encoder(x)
        # decoding
        x_prime = vae_model.decoder(z)
        # reconstruction error by binary crossentropy loss
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(x, x_prime) * 28 * 28
        )
        # KL divergence
        kld = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        if vae_model.constr_variable == "kld":
            loss = reconstruction_loss  # optimise for reconstruction err only
        elif vae_model.constr_variable == "reconstr_err":
            loss = kld  # optimise for kld only
        else:
            raise ValueError(
                f"constrained_variable must be one of ['kld', 'reconstr_err']"
            )
    # apply gradient
    grads = tape.gradient(loss, vae_model.trainable_weights)
    vae_model.optimizer.apply_gradients(zip(grads, vae_model.trainable_weights))

    # metrics log
    logits = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kld,
        "lambda": 0.0,
    }
    return logits


# Reconstruction training step (updates model params)
@tf.function
def train_w_step(x, vae_model, Lambda):
    if isinstance(x, tuple):
        x = x[0]
    with tf.GradientTape() as tape:
        # encoding
        z_mean, z_log_var, z = vae_model.encoder(x)
        # decoding
        x_prime = vae_model.decoder(z)
        # reconstruction error by binary crossentropy loss
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(x, x_prime) * 28 * 28
        )
        # KL divergence
        kld = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        # loss = lagrangian
        loss = lagrangian(vae_model, Lambda, reconstruction_loss, kld)
    # apply gradient
    grads = tape.gradient(loss, vae_model.trainable_weights)
    vae_model.optimizer.apply_gradients(zip(grads, vae_model.trainable_weights))

    # metrics log
    logits = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kld,
        "lambda": Lambda,
    }
    return logits


# Constraint training step (updates lambda). Pass optimizer.
@tf.function
def train_lambda_step(x, vae_model, opt, Lambda):
    if isinstance(x, tuple):
        x = x[0]
    with tf.GradientTape() as tape:
        # encoding
        z_mean, z_log_var, z = vae_model.encoder(x)
        # decoding
        x_prime = vae_model.decoder(z)
        # reconstruction error by binary crossentropy loss
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(x, x_prime) * 28 * 28
        )
        # KL divergence
        kld = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        # loss = - lagrangian (SGA)
        loss = -lagrangian(vae_model, Lambda, reconstruction_loss, kld)
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
    }
    return logits


# Training loop
def train_model(
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
):
    # warmup
    t_warm_0 = time.time()
    for step, train_image_batch in enumerate(dataset):
        logits = warmup_step(train_image_batch, vae_model)
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
            logits = train_lambda_step(
                train_image_batch,
                vae_model,
                opt_lambda,
                Lambda,
            )
            steps_lambda += 1
            # log
            log(logits, training_logs)

            # perform l SGD steps for model params
            for i in range(l):
                try:
                    step, train_image_batch = next(enum_data)
                except StopIteration:
                    break
                logits = train_w_step(
                    train_image_batch,
                    vae_model,
                    Lambda,
                )
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
import tensorflow as tf
from tensorflow import keras

import time
import numpy as np

from helpers import log, report

# Reconstruction training step (updates model params)
@tf.function
def train_step(x, vae_model):
    if isinstance(x, tuple):
        x = x[0]
    with tf.GradientTape() as tape:
        # encoding
        z_mean, z_log_var, z = vae_model.encoder(x)
        # decoding
        x_prime = vae_model.decoder(z)
        # reconstruction error by binary crossentropy loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(x, x_prime), axis=(1, 2))
        )
        # KL divergence
        kld = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        # loss = reconstruction error + KL divergence
        loss = reconstruction_loss + vae_model.beta * (vae_model.M / vae_model.N) * kld
    # apply gradient
    grads = tape.gradient(loss, vae_model.trainable_weights)
    vae_model.optimizer.apply_gradients(zip(grads, vae_model.trainable_weights))

    # metrics log
    logits = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kld,
    }
    return logits


# training loop
def train_model(vae_model, dataset, batch_size, epochs, training_logs):
    for epoch in range(epochs):
        t0 = time.time()
        print(f"\nStart of epoch {epoch + 1}")

        # iterate over batches
        enum_data = enumerate(dataset)
        for step, elem in enum_data:
            train_image_batch = tf.reshape(elem["image"], [batch_size, 64, 64])
            logits = train_step(train_image_batch, vae_model)
            # log
            log(logits, training_logs)

            # Log every 200 batches.
            if step % 200 == 0:
                report(batch_size, step, logits)
        t1 = time.time()
        epoch_time = t1 - t0
        training_logs["epoch_times"].append(epoch_time)
        print(f"\nEpoch time: {epoch_time:.2f}s")

    mean_epoch_time = np.mean(training_logs["epoch_times"])
    print(f"\nTraining complete! Mean epoch time: {mean_epoch_time:.2f}s")
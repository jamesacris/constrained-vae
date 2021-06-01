import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import time
from pathlib import Path
import numpy as np
import random as python_random

# sampling z with (z_mean, z_log_var)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# encoder
def create_encoder(latent_dim):
    image_input = keras.Input(shape=(64, 64, 1))
    x = layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same")(image_input)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_output = Sampling()((z_mean, z_log_var))
    encoder = keras.Model(image_input, (z_mean, z_log_var, z_output))
    return encoder

# decoder
def create_decoder(latent_dim):
    z_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(4096, activation="tanh")(z_input)
    x = layers.Reshape((64, 64, 1))(x)
    x = layers.Conv2DTranspose(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, kernel_size=(5, 5), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(1, kernel_size=(5, 5), activation="sigmoid", padding="same")(x)
    image_output = layers.Reshape((64, 64))(x)
    decoder = keras.Model(z_input, image_output)
    return decoder
    
# train step
# why using this wrapper around train_step:
# https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844
def create_train_step_func_instance():
    @tf.function
    def train_step(x, encoder, decoder, normalized_beta):
        with tf.GradientTape(persistent=True) as tape:
            # encoding
            z_mean, z_log_var, z = encoder(x)
            # decoding
            x_prime = decoder(z)
            # reconstruction loss -- Must be pixel mean for normalized beta
            reconstr_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime))
            # KL loss -- Must be latent mean for normalized beta
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # loss = reconstr_loss + normalized_beta * kl_loss
            loss = reconstr_loss + normalized_beta * kl_loss
        # compute gradients
        encoder_grads = tape.gradient(loss, encoder.trainable_weights)
        decoder_grads = tape.gradient(loss, decoder.trainable_weights)
        del tape  # a persistent tape must be deleted manually 
        # apply gradients
        encoder.optimizer.apply_gradients(zip(encoder_grads, encoder.trainable_weights))
        decoder.optimizer.apply_gradients(zip(decoder_grads, decoder.trainable_weights))
        # return losses for logging
        return loss, reconstr_loss, kl_loss
    return train_step
    
# beta VAE for dsprites
class DspritesBetaVAE():
    def __init__(self, latent_dim, normalized_beta, random_seed=0):
        self.latent_dim = latent_dim
        self.normalized_beta = normalized_beta
        self.random_seed = random_seed
        self.encoder = None
        self.decoder = None
        
    def train_save(self, dataset, epochs=10, batch_size=256, lr=.01, save_dir=None,
                   verbose_batch=100, verbose_epoch=1, batch_limit_for_debug=None):
        # batch dataset
        dataset = dataset.unbatch().batch(batch_size)
        
        # initialize NN with given seed
        python_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        self.encoder = create_encoder(self.latent_dim)
        self.decoder = create_decoder(self.latent_dim)
        
        # compile
        self.encoder.compile(optimizer=keras.optimizers.Adagrad(learning_rate=lr))
        self.decoder.compile(optimizer=keras.optimizers.Adagrad(learning_rate=lr))
        
        # training history
        hist_loss = []
        hist_reconstr_loss = []
        hist_kl_loss = []
        hist_wtime = []
        
        # train_step
        train_step_func = create_train_step_func_instance()
        
        # training loop
        t0 = time.time()
        nbatch_epoch = 'unknown'
        for epoch in range(epochs):
            for ibatch, batch in enumerate(dataset):
                # train
                image_batch = tf.squeeze(batch['image'])
                loss, reconstr_loss, kl_loss = train_step_func(image_batch, 
                    self.encoder, self.decoder, self.normalized_beta)
                
                # log history
                hist_loss.append(loss)
                hist_reconstr_loss.append(reconstr_loss)
                hist_kl_loss.append(kl_loss)
                wtime = time.time() - t0
                hist_wtime.append(wtime)
                
                # verbose
                if ibatch % verbose_batch == 0:
                    print(f'Batch {ibatch + 1} / {nbatch_epoch}: '
                        f'loss={loss.numpy():.4e}, '
                        f'reconstr_loss={reconstr_loss.numpy():.4e}, '
                        f'kl_loss={kl_loss.numpy():.4e}, '
                        f'wtime={wtime:.1f}' + (' ' * 10), end='\r')
                
                # quick debug
                if ibatch + 1 == batch_limit_for_debug:
                    break        
                
            # verbose
            nbatch_epoch = ibatch + 1
            if epoch % verbose_epoch == 0:
                print(f'Epoch {epoch + 1} / {epochs}: '
                    f'loss={loss.numpy():.4e}, '
                    f'reconstr_loss={reconstr_loss.numpy():.4e}, '
                    f'kl_loss={kl_loss.numpy():.4e}, '
                    f'wtime={wtime:.1f}' + (' ' * 10))
                    
        # save results
        if save_dir is None:
            save_dir = f'output_train/'
            save_dir += f'nlat={self.latent_dim}__'
            save_dir += f'beta={self.normalized_beta}__'
            save_dir += f'seed={self.random_seed}'
        path = Path(save_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        
        # save network hyperparameters
        with open(path / 'hypar_network.txt', 'w') as f:
            f.write(str({
                'latent_dim': self.latent_dim, 
                'normalized_beta': self.normalized_beta,
                'random_seed': self.random_seed}))
            
        # save training hyperparameters
        with open(path / 'hypar_training.txt', 'w') as f:
            f.write(str({'epochs': epochs, 'batch_size': batch_size, 'lr': lr}))
        
        # save history
        np.savetxt(path / 'hist_loss.txt', tf.concat(hist_loss, axis=0).numpy())
        np.savetxt(path / 'hist_reconstr_loss.txt', tf.concat(hist_reconstr_loss, axis=0).numpy())
        np.savetxt(path / 'hist_kl_loss.txt', tf.concat(hist_kl_loss, axis=0).numpy())
        np.savetxt(path / 'hist_wtime.txt', np.array(hist_wtime))

        # save model weights
        self.encoder.save_weights(path / 'weights_encoder.h5')
        self.decoder.save_weights(path / 'weights_decoder.h5')
        
    def load_model_weights(self, weights_encoder_h5, weights_decoder_h5):
        # create model
        self.encoder = create_encoder(self.latent_dim)
        self.decoder = create_decoder(self.latent_dim)
        # load model
        self.encoder.load_weights(weights_encoder_h5)
        self.decoder.load_weights(weights_decoder_h5)

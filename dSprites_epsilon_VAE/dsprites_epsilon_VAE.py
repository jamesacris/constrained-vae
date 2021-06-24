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
def create_encoder(latent_dim, n_filters_first_conv2d):
    image_input = keras.Input(shape=(64, 64, 1))
    x = layers.Conv2D(n_filters_first_conv2d, kernel_size=(5, 5), activation="relu", padding="same")(image_input)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(n_filters_first_conv2d * 2, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_output = Sampling()((z_mean, z_log_var))
    encoder = keras.Model(image_input, (z_mean, z_log_var, z_output))
    return encoder

# decoder
def create_decoder(latent_dim, n_filters_first_conv2d):
    z_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(4096, activation="tanh")(z_input)
    x = layers.Reshape((64, 64, 1))(x)
    x = layers.Conv2DTranspose(n_filters_first_conv2d * 2, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(n_filters_first_conv2d, kernel_size=(5, 5), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(1, kernel_size=(5, 5), activation="sigmoid", padding="same")(x)
    image_output = layers.Reshape((64, 64))(x)
    decoder = keras.Model(z_input, image_output)
    return decoder

# lagrange multiplier lambda (a trainable parameter)
# TODO: Is there a better way to wrap this up, e.g. a small keras model similar to encoder/decoder?
#       Passing optimiser to train_lambda_step is awkward.
def create_lambda_with_optimizer(initial_learning_rate=0.01,
    decay_steps=1, decay_rate=1e-3):
    Lambda = tf.Variable(0.0)
    learning_rate_lambda = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate, decay_steps, decay_rate
    )
    opt_lambda = keras.optimizers.SGD(learning_rate=learning_rate_lambda)
    return Lambda, opt_lambda

# train steps
# why using this wrapper around train_step:
# https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844
# Lagrangian loss function
def create_lagrangian_func_instance():
    @tf.function
    def lagrangian(epsilon, constr_variable, Lambda, reconstr_loss, kld):
        # constrain kld
        if constr_variable == "kld":
            # constraint h
            h = tf.nn.relu(kld - epsilon)
            # Lagrangian
            l = reconstr_loss + Lambda * h
        # constrain reconstruction error
        elif constr_variable == "reconstr_err":
            # constraint h
            h = tf.nn.relu(reconstr_loss - epsilon)
            # Lagrangian
            l = kld + Lambda * h
        else:
            raise ValueError(f"constrained_variable must be one of ['kld', 'reconstr_err']")
        return tf.reduce_mean(l)
    return lagrangian

# Warmup training step (this is just train_w_step with lambda = 0)
def create_warmup_step_func_instance():
    @tf.function
    def warmup_step(x, encoder, decoder, constr_variable):
        with tf.GradientTape(persistent=True) as tape:
            # encoding
            z_mean, z_log_var, z = encoder(x)
            # decoding
            x_prime = decoder(z)
            # reconstruction error -- Must be pixel mean for normalized epsilon
            reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime))
            # KL divergence -- Must be latent mean for normalized epsilon
            kld = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            if constr_variable == "kld":
                loss = reconstruction_loss  # optimise for reconstruction err only
            elif constr_variable == "reconstr_err":
                loss = kld  # optimise for kld only
            else:
                raise ValueError(
                    f"constrained_variable must be one of ['kld', 'reconstr_err']"
                )
        # compute gradients
        encoder_grads = tape.gradient(loss, encoder.trainable_weights)
        decoder_grads = tape.gradient(loss, decoder.trainable_weights)
        del tape  # a persistent tape must be deleted manually 
        # apply gradients
        encoder.optimizer.apply_gradients(zip(encoder_grads, encoder.trainable_weights))
        decoder.optimizer.apply_gradients(zip(decoder_grads, decoder.trainable_weights))
        # return losses for logging
        return loss, reconstruction_loss, kld
    return warmup_step

# weights training step (updates model params)
def create_train_weights_step_func_instance():
    @tf.function
    def train_w_step(x, Lambda, constr_variable, epsilon, encoder, decoder, lagrangian):
        with tf.GradientTape(persistent=True) as tape:
            # encoding
            z_mean, z_log_var, z = encoder(x)
            # decoding
            x_prime = decoder(z)
            # reconstruction error -- Must be pixel mean for normalized epsilon
            reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime))
            # KL divergence -- Must be latent mean for normalized epsilon
            kld = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # loss = lagrangian
            loss = lagrangian(epsilon, constr_variable, Lambda, reconstruction_loss, kld)
        # compute gradients
        encoder_grads = tape.gradient(loss, encoder.trainable_weights)
        decoder_grads = tape.gradient(loss, decoder.trainable_weights)
        del tape  # a persistent tape must be deleted manually 
        # apply gradients
        encoder.optimizer.apply_gradients(zip(encoder_grads, encoder.trainable_weights))
        decoder.optimizer.apply_gradients(zip(decoder_grads, decoder.trainable_weights))
        # return losses for logging
        return loss, reconstruction_loss, kld, Lambda
    return train_w_step

# Constraint training step (updates lambda). Pass optimizer.
def create_train_lambda_step_func_instance():
    @tf.function
    def train_lambda_step(x, Lambda, lambda_opt, constr_variable, epsilon, encoder, decoder, lagrangian):
        with tf.GradientTape(persistent=True) as tape:
            # encoding
            z_mean, z_log_var, z = encoder(x)
            # decoding
            x_prime = decoder(z)
            # reconstruction error -- Must be pixel mean for normalized epsilon
            reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime))
            # KL divergence -- Must be latent mean for normalized epsilon
            kld = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # loss = - lagrangian (SGA)
            loss = -lagrangian(epsilon, constr_variable, Lambda, reconstruction_loss, kld)
        # calculate and apply gradient
        grad = tape.gradient(target=loss, sources=[Lambda])
        del tape  # a persistent tape must be deleted manually
        # opt = tf.keras.optimizers.Adam()
        lambda_opt.apply_gradients(zip(grad, [Lambda]))
        # return losses for logging
        return loss, reconstruction_loss, kld, Lambda
    return train_lambda_step


# epsilon VAE for dsprites
class DspritesEpsilonVAE():
    def __init__(self, normalized_epsilon, constrained_variable,
        latent_dim, n_filters_first_conv2d, random_seed=0):
        self.normalized_epsilon = normalized_epsilon
        self.constrained_variable = constrained_variable
        self.latent_dim = latent_dim
        self.n_filters_first_conv2d = n_filters_first_conv2d
        self.random_seed = random_seed
        self.encoder = None
        self.decoder = None
        self.lambda_optimizer = None
        
    def train_save(self, dataset, epochs=10, warmup_iters=100, l=1, d=1, nd=2, batch_size=256, lr=.01, save_dir=None,
                   verbose_batch=100, verbose_warmup=True, verbose_epoch=1, batch_limit_for_debug=None):
        # batch dataset
        dataset = dataset.unbatch().batch(batch_size)
        
        # initialize NN with given seed
        python_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        self.encoder = create_encoder(self.latent_dim, self.n_filters_first_conv2d)
        self.decoder = create_decoder(self.latent_dim, self.n_filters_first_conv2d)
        
        # compile
        self.encoder.compile(optimizer=keras.optimizers.Adagrad(learning_rate=lr))
        self.decoder.compile(optimizer=keras.optimizers.Adagrad(learning_rate=lr))
        
        # training history
        hist_loss = []
        hist_reconstr_loss = []
        hist_kl_loss = []
        hist_wtime = []
        # lambda
        hist_lambda = []
        hist_lr_lambda = []
        
        # instantiate lambda and optimizer
        Lambda, lambda_opt = create_lambda_with_optimizer()

        # instantiate training step functions
        lagrangian = create_lagrangian_func_instance()
        warmup_step = create_warmup_step_func_instance()
        train_w_step = create_train_weights_step_func_instance()
        train_lambda_step = create_train_lambda_step_func_instance()
        
        # warmup loop (train without constraint)
        # TODO Where to output warmup time to?
        t0 = time.time()
        for ibatch, batch in enumerate(dataset):
            # warmup
            image_batch = tf.squeeze(batch['image'])
            loss, reconstr_loss, kl_loss = warmup_step(image_batch,
            self.encoder, self.decoder, self.constrained_variable)

            # log history
            hist_loss.append(loss)
            hist_reconstr_loss.append(reconstr_loss)
            hist_kl_loss.append(kl_loss)
            wtime = time.time() - t0
            hist_wtime.append(wtime)
            hist_lambda.append(Lambda)
            hist_lr_lambda.append(lambda_opt._decayed_lr('float32').numpy())

            # verbose
            if verbose_batch > 0 and ibatch % verbose_batch == 0:
                print(f'Warmup batch {ibatch + 1} / {warmup_iters}: '
                    f'loss={loss.numpy():.4e}, '
                    f'reconstr_loss={reconstr_loss.numpy():.4e}, '
                    f'kl_loss={kl_loss.numpy():.4e}, '
                    f'wtime={wtime:.1f}' + (' ' * 10), end='\r')
            
            # stop when warmup complete
            if ibatch + 1 >= warmup_iters:
                break
        
        # verbose
        if verbose_warmup:
            print(f'Warmup: '
                f'loss={loss.numpy():.4e}, '
                f'reconstr_loss={reconstr_loss.numpy():.4e}, '
                f'kl_loss={kl_loss.numpy():.4e}, '
                f'wtime={wtime:.1f}' + (' ' * 10))

        # training loop (train with constraint)
        t0 = time.time()
        nbatch_epoch = 'unknown'
        for epoch in range(epochs):
            enum_data = enumerate(dataset)
            for ibatch, batch in enum_data:
                # perform one 'train lambda' step
                image_batch = tf.squeeze(batch['image'])
                loss, reconstr_loss, kl_loss, Lambda = train_lambda_step(
                    image_batch, Lambda, lambda_opt, self.constrained_variable,
                    self.normalized_epsilon, self.encoder, self.decoder, lagrangian)
                
                # log history
                hist_loss.append(loss)
                hist_reconstr_loss.append(reconstr_loss)
                hist_kl_loss.append(kl_loss)
                wtime = time.time() - t0
                hist_wtime.append(wtime)
                hist_lambda.append(Lambda)
                hist_lr_lambda.append(lambda_opt._decayed_lr('float32').numpy())
                
                # verbose
                if verbose_batch > 0 and ibatch % verbose_batch == 0:
                    print(f'Training batch {ibatch + 1} / {nbatch_epoch}: '
                        f'loss={loss.numpy():.4e}, '
                        f'reconstr_loss={reconstr_loss.numpy():.4e}, '
                        f'kl_loss={kl_loss.numpy():.4e}, '
                        f'lambda={Lambda.numpy():.4e}, '
                        f'wtime={wtime:.1f}' + (' ' * 10), end='\r')
                
                # perform l 'train model weight' steps
                for i in range(l):
                    # attempt to get next batch, if there is one
                    try:
                        ibatch, batch = next(enum_data)
                    except StopIteration:
                        break

                    # train model weights
                    image_batch = tf.squeeze(batch['image'])
                    loss, reconstr_loss, kl_loss, Lambda = train_w_step(
                        image_batch, Lambda, self.constrained_variable,
                        self.normalized_epsilon, self.encoder, self.decoder, lagrangian)

                    # verbose
                    if verbose_batch > 0 and ibatch % verbose_batch == 0:
                        print(f'Training batch {ibatch + 1} / {nbatch_epoch}: '
                            f'loss={loss.numpy():.4e}, '
                            f'reconstr_loss={reconstr_loss.numpy():.4e}, '
                            f'kl_loss={kl_loss.numpy():.4e}, '
                            f'lambda={Lambda.numpy():.4e}, '
                            f'wtime={wtime:.1f}' + (' ' * 10), end='\r')

                # update l every nd-th step
                if ibatch % nd == 0:
                    l = l + d

                # quick debug
                if ibatch + 1 == batch_limit_for_debug:
                    break        
                
            # verbose
            nbatch_epoch = ibatch + 1
            if verbose_epoch > 0 and epoch % verbose_epoch == 0:
                print(f'Epoch {epoch + 1} / {epochs}: '
                    f'loss={loss.numpy():.4e}, '
                    f'reconstr_loss={reconstr_loss.numpy():.4e}, '
                    f'kl_loss={kl_loss.numpy():.4e}, '
                    f'lambda={Lambda.numpy():.4e}, '
                    f'wtime={wtime:.1f}' + (' ' * 10))
                    
        # save results
        if save_dir is None:
            save_dir = f'output_train/'
            save_dir += f'constr={self.constrained_variable}__'
            save_dir += f'epsilon={self.normalized_epsilon}__'
            save_dir += f'nlat={self.latent_dim}__'
            save_dir += f'nConv2D={self.n_filters_first_conv2d}__'
            save_dir += f'seed={self.random_seed}'
        path = Path(save_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        
        # save network hyperparameters
        with open(path / 'hypar_network.txt', 'w') as f:
            f.write(str({
                'constrained_variable': self.constrained_variable,
                'normalized_epsilon': self.normalized_epsilon,
                'latent_dim': self.latent_dim, 
                'n_filters_first_conv2d': self.n_filters_first_conv2d,
                'random_seed': self.random_seed}))
            
        # save training hyperparameters
        with open(path / 'hypar_training.txt', 'w') as f:
            f.write(str({
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'warmup_iters': warmup_iters,
                'l': l,
                'd': d,
                'nd': nd}))
        
        # save history
        np.savetxt(path / 'hist_loss.txt', tf.concat(hist_loss, axis=0).numpy())
        np.savetxt(path / 'hist_reconstr_loss.txt', tf.concat(hist_reconstr_loss, axis=0).numpy())
        np.savetxt(path / 'hist_kl_loss.txt', tf.concat(hist_kl_loss, axis=0).numpy())
        np.savetxt(path / 'hist_wtime.txt', np.array(hist_wtime))
        np.savetxt(path / 'hist_lambda.txt', tf.concat(hist_lambda, axis=0).numpy())
        np.savetxt(path / 'hist_lr_lambda.txt', np.array(hist_lr_lambda))

        # save model weights
        self.encoder.save_weights(path / 'weights_encoder.h5')
        self.decoder.save_weights(path / 'weights_decoder.h5')
        
    def load_model_weights(self, weights_encoder_h5, weights_decoder_h5):
        # create model
        self.encoder = create_encoder(self.latent_dim, self.n_filters_first_conv2d)
        self.decoder = create_decoder(self.latent_dim, self.n_filters_first_conv2d)
        # load model
        self.encoder.load_weights(weights_encoder_h5)
        self.decoder.load_weights(weights_decoder_h5)

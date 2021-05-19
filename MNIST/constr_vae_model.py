import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# sampling z with (z_mean, z_log_var)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# latent dimension
latent_dim = 2

# build the encoder (convolutional layers)
image_input = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(8, kernel_size=(4, 4), activation="relu", padding="same")(image_input)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same")(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z_output = Sampling()([z_mean, z_log_var])
encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output], name="encoder")
encoder_VAE.summary()

# build the decoder (dense layers)
z_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(16, activation="relu")(z_input)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(28 * 28, activation="sigmoid")(x)
image_output = layers.Reshape((28, 28))(x)
decoder_VAE = keras.Model(z_input, image_output, name="decoder")
decoder_VAE.summary()

# Constrained VAE class
class constr_VAE(keras.Model):
    # constructor
    # remove beta, add in KLD_aim,
    def __init__(self, encoder, decoder, epsilon, constr_variable, **kwargs):
        super(constr_VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.epsilon = epsilon
        self.constr_variable = constr_variable
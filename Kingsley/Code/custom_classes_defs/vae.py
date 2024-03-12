"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2023/11/22
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
Accelerator: GPU

Adapted and modified:
By : Bawfeh Kingsley Kometa
Date: 08/03/2024
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt



"""
## Create a sampling layer
"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]


    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs, training=True)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(targets, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        inputs, targets = data
        z_mean, z_log_var, z = self.encoder(inputs, training=False)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(targets, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }


class AE_variational(VAE):
    def __init__(self, in_shape, panel_size, latent_dim=32,
                 kernel_size=3, pool_size=2, ARCH_NN='vae', **kwargs):
        super().__init__(encoder=None, decoder=None, **kwargs)
        self.latent_dim = latent_dim
        self.ARCH_NN = ARCH_NN.lower()
        self.in_shape = in_shape
        self.panel_size = panel_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.depth = len(panel_size) \
              if isinstance(panel_size, list) else panel_size
        
        # input image
        inputs = keras.Input(shape=self.in_shape, name='inputs')

        # encoder
        self.encoder = keras.Model(inputs, self.get_encoder(inputs), name='encoder')
        
        # autoencoder
        self.autoencoder = keras.Model(inputs, self.get_decoder(self.encoder.output[-1]),
                                       name='autoencoder')

        # decoder
        decoder_inputs = keras.Input(shape=self.encoder.output_shape[-1][1:], name='codes')
        latent_layer = np.nonzero([s.name=='latents' for s in self.autoencoder.layers])[0][0]

        for i, layer in enumerate(self.autoencoder.layers[latent_layer+1:]):
            decoder_outputs = layer(decoder_inputs if i==0 else decoder_outputs)

        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')


    def predict(self, x_test, source='images', target='images', verbose=1):

        if (source=='codes' and target=='images'):
            return self.decoder.predict(x_test, verbose=verbose)

        elif (source=='images' and target=='codes'):
            return self.encoder.predict(x_test, verbose=verbose)[-1]

        return self.autoencoder.predict(x_test, verbose=verbose) 


    def get_encoder(self, inputs):
        """ Creates encoder model for VAE architecture """
    
        for i in range(self.depth):
            level = self.depth-i-1
            encoded = layers.Conv2D(self.panel_size[i], self.kernel_size, 
                              activation='relu', padding='same',
                               name='Conv2D_%i'%level)(inputs if i==0 else encoded)
            assert encoded.shape[1] >= self.kernel_size
            encoded = layers.MaxPooling2D(self.pool_size, 
                                    padding='same', name='encode_%i'%level)(encoded)
            if np.mod(encoded.shape[1], self.pool_size):
                self.depth = i+1  # reset network dimension
                break
        self.encoded_shape = encoded.shape[1:]
        x = layers.Flatten(name='flatten')(encoded)
        x = layers.Dense(self.latent_dim // 2, activation="relu", name='dense_codes_0')(x)
            
        if self.ARCH_NN == 'vae':
            z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
            z = Sampling(name='latents')([z_mean, z_log_var])
        else: # (vanilla) autoencoder 
            z = layers.Dense(self.latent_dim, name="latents")(x)
            z_log_var = tf.zeros_like(z)
            z_mean = tf.zeros_like(z)
            
            
        return z_mean, z_log_var, z
    

    def get_decoder(self, latent_inputs):
        """ Creates graph for decoder model """
    
        decoded = layers.Dense(np.prod(self.encoded_shape), activation="relu",
                               name='dense_codes_1')(latent_inputs)
        decoded = layers.Reshape(self.encoded_shape)(decoded)
        
        for i in range(self.depth):
            decoded = layers.Conv2DTranspose(self.panel_size[self.depth-i-1], self.kernel_size, 
                              activation='relu', padding='same', 
                              name='Conv2DT_%i'%i)(decoded)

            assert decoded.shape[1] >= self.pool_size
            decoded = layers.UpSampling2D(self.pool_size, name='decode_%i'%i)(decoded)

        decoded = layers.Conv2DTranspose(self.encoder.input_shape[-1], self.kernel_size,
                                         activation="sigmoid", padding="same", name='outputs')(decoded)
    
        return decoded


def plot_latent_space(vae, n=30, figsize=15):
    """display a n*n 2D manifold of digits"""
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

def plot_label_clusters(vae, data, labels):
    """display a 2D plot of the digit classes in the latent space"""
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


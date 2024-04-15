from custom_classes_defs.setup import *

## ===================================================
class AE_dense(model_config):
    """ 
    Autoencoder class using dense layers. 
    An organized V-cycle of layers: 
    - dense layers consist of multiples of 2 the latent dimension
    """
    def __init__(self, depth, latent_dim=32, model_arch=None, **kwargs):
        """
        @params:
        - latent_dim = encoder dimension
        - img_shape = shape of input data sample
        - depth = number of dense layers used for defining the encoder
        """
        super().__init__(**kwargs)
        self.latent_dim = latent_dim 
        self.depth = depth 
        self.update_model_arch(model_arch)

        # input image
        inputs = keras.Input(shape=self.img_shape+(self.channels_dim[0],), name='inputs')
        
        # encoder
        self.encoder = keras.Model(inputs, self.encoder_block(inputs), name='encoder')
        
        # autoencoder
        self.autoencoder = keras.Model(inputs, self.decoder_block(self.encoder.output),
                                       name='autoencoder')
        # decoder
        decoder_inputs = keras.Input(shape=self.encoder.output_shape[1:], name='codes')
        latent_layer = np.nonzero([s.name=='encode_0' for s in self.autoencoder.layers])[0][0]

        for i, layer in enumerate(self.autoencoder.layers[latent_layer+1:]):
            decoder_outputs = layer(decoder_inputs if i==0 else decoder_outputs)

        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
        
    def encoder_block(self, inputs):
        """ returns the graph used to define an encoder """
        dim = self.latent_dim * np.power(2, range(self.depth-1,-1,-1))
        inputs = self.take_off(inputs)
        flat_inputs = layers.Flatten()(inputs)
        for i in range(self.depth):
            level = self.depth-i-1
            encoded = layers.Dense(dim[i], activation='relu', name='encode_%i'%level) \
                    (flat_inputs if i==0 else encoded)
        return encoded

    def decoder_block(self, inputs):
        """ returns the graph used to define a decoder """
        dim = self.latent_dim * np.power(2, range(self.depth))
        for i in range(1,self.depth):
            decoded = layers.Dense(dim[i], activation='relu', name='decode_%i'%i)\
                            (inputs if i==1 else decoded)
        # output image       
        decoded = layers.Dense(np.prod(self.target_size)*self.channels_dim[1], 
                               activation='sigmoid', name='outputs')\
                    (decoded if self.depth>1 else inputs)

        decoded = layers.Reshape(self.target_size+(self.channels_dim[1],))(decoded)

        decoded = self.landing(decoded)

        return decoded


## ===================================================
class AE_convolutional(model_config):
    """
    Autoencoder class using convolutional layers:
        - An organized V-cycle of layers
        - A list of panel sizes determine the depth of the network
    """
    
    def __init__(self, panel_size, kernel_size=3, pool_size=2, model_arch=None, **kwargs):
        """
        @params:
            - img_shape = shape of input data sample
            - panel_size = list of panel sizes or an int if only network depth=1
            - kernel_size = dimension of convolutional filter/kernel
            - pool_size = max-pool dimension for up/downsampling 
        """
        super().__init__(**kwargs)
        self.panel_size = panel_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.depth = len(panel_size) \
              if isinstance(panel_size,list) else panel_size

        self.update_model_arch(model_arch)
        
        # encoder
        inputs = keras.Input(shape=self.img_shape+(self.channels_dim[0],), name='inputs')  # input image
        self.encoder = keras.Model(inputs, self.encoder_block(inputs), name='encoder')
        
        # autoencoder
        decoded = self.decoder_block(self.encoder.output)
        self.autoencoder = keras.Model(inputs, decoded, name='autoencoder')

        # decoder
        decoder_inputs = keras.Input(shape=self.encoder.output_shape[1:], name='codes')
        num_decoder_layers = len(self.autoencoder.layers) - len(self.encoder.layers)
        decoder_outputs = self.autoencoder.layers[-num_decoder_layers](decoder_inputs)
        if num_decoder_layers>1:
            for layer in self.autoencoder.layers[-num_decoder_layers+1:]:
                decoder_outputs = layer(decoder_outputs)

        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    def encoder_block(self, inputs):
        """ returns the graph used to define an encoder """
        inputs = self.take_off(inputs)
        for i in range(self.depth):
            level = self.depth-i-1
            encoded = layers.Conv2D(self.panel_size[i], self.kernel_size, 
                              activation='relu', padding='same',
                               name='Conv2D_%i'%level)(inputs if i==0 else encoded)
            assert encoded.shape[1] >= self.kernel_size

            encoded = layers.MaxPooling2D(self.pool_size, 
                                    padding='same', name='encode_%i'%level)(encoded)

            if np.mod(encoded.shape[1], self.pool_size):
                # discontinue adding of layers 
                # if dimension of ouput layer is not compatible with pooling dimension
                self.depth = i+1  # reset network dimension
                break

        return encoded

    def decoder_block(self, inputs):
        """ returns the graph used to define a decoder """
        for i in range(self.depth):
            decoded = layers.Conv2D(self.panel_size[self.depth-i-1], self.kernel_size, 
                              activation='relu', padding='same', 
                              name='Conv2DT_%i'%i)(inputs if i==0 else decoded)

            assert decoded.shape[1] >= self.pool_size

            decoded = layers.UpSampling2D(self.pool_size, name='decode_%i'%i)(decoded)

        decoded = layers.Conv2D(self.channels_dim[1], self.kernel_size, activation='sigmoid',
                                padding='same', name='outputs')(decoded)

        decoded = self.landing(decoded)

        return decoded


## ===================================================
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



## ===================================================
class AE_variational(model_config):
    def __init__(self, panel_size, latent_dim=32,
                 kernel_size=3, pool_size=2, ARCH_NN='vae', model_arch=None, **kwargs):
        super().__init__(**kwargs)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.latent_dim = latent_dim
        self.ARCH_NN = ARCH_NN.lower()
        self.panel_size = panel_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.depth = len(panel_size) \
              if isinstance(panel_size,list) else panel_size

        self.update_model_arch(model_arch)
       
        # input image
        inputs = keras.Input(shape=self.img_shape+(self.channels_dim[0],), name='inputs')

        # encoder
        self.encoder = keras.Model(inputs, self.encoder_block(inputs), name='encoder')
        
        # autoencoder
        self.autoencoder = keras.Model(inputs, self.decoder_block(self.encoder.output[-1]),
                                       name='autoencoder')

        # decoder
        decoder_inputs = keras.Input(shape=self.encoder.output_shape[-1][1:], name='codes')
        latent_layer = np.nonzero([s.name=='latents' for s in self.autoencoder.layers])[0][0]

        for i, layer in enumerate(self.autoencoder.layers[latent_layer+1:]):
            decoder_outputs = layer(decoder_inputs if i==0 else decoder_outputs)

        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    def encoder_block(self, inputs):
        """ Creates graph need for encoder model """
    
        inputs = self.take_off(inputs)
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
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
            
        if self.ARCH_NN == 'vae':
            z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
            z = Sampling(name='latents')([z_mean, z_log_var])
        else: # (vanilla) autoencoder 
            z_mean = layers.Dense(self.latent_dim, trainable=False, name="z_mean")(x)
            z_log_var = layers.Dense(self.latent_dim, trainable=False, name="z_log_var")(x)
            z = layers.Dense(self.latent_dim, name="latents")(x)
            
        return z_mean, z_log_var, z
    

    def decoder_block(self, latent_inputs):
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

        decoded = layers.Conv2DTranspose(self.channels_dim[1], self.kernel_size,
                                         activation="sigmoid", padding="same", name='outputs')(decoded)
        decoded = self.landing(decoded)
    
        return decoded

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
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
            "kl_loss": self.kl_loss_tracker.result(),
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
            "kl_loss": self.kl_loss_tracker.result(),
        }


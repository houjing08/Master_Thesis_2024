import tensorflow as tf
import keras
from keras import layers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, average_precision_score

class model_config(keras.Model):
    """ Define common parameters required for training any model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compile_args = dict(optimizer='adam', loss='binary_crossentropy')
        self.training_args = dict(epochs=1, batch_size=1, shuffle=True, verbose=0)
    
    def config(self, conf_type='train', **kwargs):
        """ Resets default model compile and fit parameters """
        if conf_type=='compile':
            for key,value in kwargs.items():
                self.compile_args[key] = value
        else: # 'train'
            for key,value in kwargs.items():
                self.training_args[key] = value


    def predict(self, x_test, source='images', target='images'):

        if (source=='codes' and target=='images'):
            return self.decoder.predict(x_test)

        elif (source=='images' and target=='codes'):
            return self.encoder.predict(x_test)

        return self.autoencoder.predict(x_test) 

    def evaluate_sklearn(self, x, y, threshold=0.9, report=False):
        """ 
        Metrics to evaluate model performance:
        @params:
            - x = original images
            - y = predicted images
            - threshold = probability threshold used to define pixel labels
            - report = flag to print a report summary (according to scikit-learn)
        @returns:
            - AP score (Average precision = area under precision-recall curve)
            - ODS score (F1-score based on global thresholding)
            - OIS score (F1-score based on per image thresholding)
        """
        y_true = np.where(x>0.99, 1, 0).flatten()
        AP = average_precision_score(y_true, y.flatten())

        # ODS
        y_pred = np.where(y>threshold, 1, 0).flatten()
        reports = classification_report(y_true, y_pred)
        mvals = reports.split('\n\n')[1].split('\n')[-1].split()
        f1_ods = float(mvals[-2])

        #OIS
        y_pred = np.array([np.where(s > s.max()*threshold, 1, 0) for s in y]).flatten()
        reports = classification_report(y_true, y_pred)
        mvals = reports.split('\n\n')[1].split('\n')[-1].split()
        f1_ois = float(mvals[-2])
        
        if report:
            print(reports)

        return {'Avg-precision': np.round(AP,2), 'f1-score-ODS': f1_ods, 'f1-score-OIS': f1_ois}


    def equal(self, shape1, shape2):
        return (np.array(shape1 == shape2)).all()

## ===================================================
class AE_dense(model_config):
    """ 
    Autoencoder class using dense layers. 
    An organized V-cycle of layers: 
    - dense layers consist of multiples of 2 the latent dimension
    """
    def __init__(self, latent_dim, img_shape, depth, **kwargs):
        """
        @params:
        - latent_dim = encoder dimension
        - img_shape = shape of input data sample
        - depth = number of dense layers used for defining the encoder
        """
        super().__init__(**kwargs)
        self.latent_dim = latent_dim 
        self.img_shape = img_shape
        self.depth = depth 
        
        # input image
        inputs = keras.Input(shape=self.img_shape, name='inputs')
        
        # encoder
        self.encoder = keras.Model(inputs, self.get_encoder(inputs), name='encoder')
        
        # autoencoder
        self.autoencoder = keras.Model(inputs, self.get_decoder(self.encoder.output),
                                       name='autoencoder')
        # decoder
        decoder_inputs = keras.Input(shape=self.encoder.output_shape[1:], name='codes')
        decoder_outputs = self.autoencoder.layers[-self.depth](decoder_inputs)
        if self.depth>1:
            # loop over decoder layers of autoencoder
            for layer in self.autoencoder.layers[-self.depth+1:]:
                decoder_outputs = layer(decoder_outputs)

        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
        
    def get_encoder(self, inputs):
        """ returns the graph used to define an encoder """
        dim = self.latent_dim * np.power(2, range(self.depth-1,-1,-1))
        for i in range(self.depth):
            level = self.depth-i-1
            encoded = layers.Dense(dim[i],activation='relu', name='encode_%i'%level) \
                    (inputs if i==0 else encoded)
        return encoded

    def get_decoder(self, inputs):
        """ returns the graph used to define a decoder """
        dim = self.latent_dim * np.power(2, range(self.depth))
        for i in range(self.depth):
            decoded = layers.Dense(dim[i], activation='relu', name='decode_%i'%i)\
                            (inputs if i==0 else decoded)
        # output image       
        decoded = layers.Dense(self.img_shape[0], activation='sigmoid', name='outputs')\
                    (decoded if self.depth>1 else inputs)
        
        return decoded


## ===================================================
def display_sample_images(x_test, y, img_shape, n=10, figsize=(20,4)):
    n = 10  # How many digits we will display
    plt.figure(figsize=figsize)
    for k, i in enumerate(np.random.randint(0,x_test.shape[0],size=n)):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        ax.imshow(x_test[i].reshape(img_shape), cmap='gray')
        ax.set_axis_off()
    
        # Display reconstruction
        ax = plt.subplot(2, n, k + 1 + n)
        ax.imshow(y[i].reshape(img_shape), cmap='gray')
        ax.set_axis_off()
    plt.show()

## ===================================================
def show_convergence(history, metrics='loss'):
    """ Plot model train/validation loss history
    in order to monitor convergence over epochs """
    if isinstance(metrics, list):
        for metric in metrics:
            if metric in history.history:
                plt.plot(history.history[metric], label=metric)
            else:
                print(f'cannot find {metric} in history')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    elif metrics in history.history:
        plt.plot(history.history[metrics])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    else:
        print(f'cannot find {metrics} in history')



## ===================================================
class AE_convolutional(model_config):
    """
    Autoencoder class using convolutional layers:
        - An organized V-cycle of layers
        - A list of panel sizes determine the depth of the network
    """
    
    def __init__(self, img_shape, panel_size, kernel_size=3, pool_size=2, **kwargs):
        """
        @params:
            - img_shape = shape of input data sample
            - panel_size = list of panel sizes or an int if only network depth=1
            - kernel_size = dimension of convolutional filter/kernel
            - pool_size = max-pool dimension for up/downsampling 
        """
        super().__init__(**kwargs)
        self.img_shape = img_shape
        self.panel_size = panel_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.depth = len(panel_size) \
              if isinstance(panel_size,list) else panel_size
        
        # encoder
        inputs = keras.Input(shape=self.img_shape, name='inputs')  # input image
        self.encoder = keras.Model(inputs, self.get_encoder(inputs), name='encoder')
        
        # autoencoder
        self.autoencoder = keras.Model(inputs, self.get_decoder(self.encoder.output),
                                        name='autoencoder')
        
        # decoder
        decoder_inputs = keras.Input(shape=self.encoder.output_shape[1:], name='codes')
        num_decoder_layers = len(self.autoencoder.layers) - len(self.encoder.layers)
        decoder_outputs = self.autoencoder.layers[-num_decoder_layers](decoder_inputs)
        if num_decoder_layers>1:
            for layer in self.autoencoder.layers[-num_decoder_layers+1:]:
                decoder_outputs = layer(decoder_outputs)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    def get_encoder(self, inputs):
        """ returns the graph used to define an encoder """
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

    def get_decoder(self, inputs):
        """ returns the graph used to define a decoder """
        for i in range(self.depth):
            decoded = layers.Conv2D(self.panel_size[self.depth-i-1], self.kernel_size, 
                              activation='relu', padding='same', 
                              name='Conv2DT_%i'%i)(inputs if i==0 else decoded)

            assert decoded.shape[1] >= self.pool_size

            decoded = layers.UpSampling2D(self.pool_size, name='decode_%i'%i)(decoded)

        decoded = layers.Conv2D(1, self.kernel_size, activation='sigmoid',
                                padding='same', name='outputs')(decoded)

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
    def __init__(self, img_shape, panel_size, latent_dim=32,
                 kernel_size=3, pool_size=2, ARCH_NN='vae', **kwargs):
        super().__init__(**kwargs)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.latent_dim = latent_dim
        self.ARCH_NN = ARCH_NN.lower()
        self.img_shape = img_shape
        self.panel_size = panel_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.depth = len(panel_size) \
              if isinstance(panel_size,list) else panel_size
       
        # input image
        inputs = keras.Input(shape=self.img_shape, name='inputs')

        # encoder
        self.encoder = keras.Model(inputs, self.get_encoder(inputs), name='encoder')
        
        # autoencoder
        self.autoencoder = keras.Model(inputs, self.get_decoder(self.encoder.output[-1]),
                                       name='autoencoder')

        # decoder
        decoder_inputs = keras.Input(shape=self.encoder.output_shape[-1][1:], name='codes')
        layer_num  = len(self.encoder.layers)

        for i, layer in enumerate(self.autoencoder.layers[layer_num:]):
            decoder_outputs = layer(decoder_inputs if i==0 else decoder_outputs)

        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    def get_encoder(self, inputs):
        """ Creates graph need for encoder model """
    
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



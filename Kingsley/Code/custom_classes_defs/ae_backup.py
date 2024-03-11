import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, average_precision_score

class model_config():
    """ Define common parameters required for training any model """
    def __init__(self):
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

#         if conf_type=='compile':
#             x = self.compile_args
#         else: # 'train'
#             x = self.training_args
# 
#         if len(kwargs.items()):
#             mylist = list(x.items())
#             mylist.extend(kwargs.items())
#             if conf_type=='compile':
#                 self.compile_args = dict(mylist)
#             else:
#                 self.training_args = dict(mylist)

    def predict(self, x_test, source='images', target='images'):

        if (source=='codes' and target=='images'):
            return self.decoder.predict(x_test)

        elif (source=='images' and target=='codes'):
            return self.encoder.predict(x_test)

        return self.autoencoder.predict(x_test) 

    def evaluate(self, x, y, threshold=0.9, report=False):
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

        return (AP, f1_ods, f1_ois)


    def equal(self, shape1, shape2):
        return (np.array(shape1 == shape2)).all()

## ===================================================
class AE_dense(model_config):
    """ 
    Autoencoder class using dense layers. 
    An organized V-cycle of layers: 
    - dense layers consist of multiples of 2 the latent dimension
    """
    def __init__(self, latent_dim, input_shape, depth):
        """
        @params:
        - latent_dim = encoder dimension
        - input_shape = shape of input data sample
        - depth = number of dense layers used for defining the encoder
        """
        super().__init__()
        self.latent_dim = latent_dim 
        self.input_shape = input_shape
        self.depth = depth 
        
        # input image
        inputs = keras.Input(shape=self.input_shape, name='inputs')
        
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
        decoded = layers.Dense(self.input_shape[0], activation='sigmoid', name='outputs')\
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
    
    def __init__(self, input_shape, panel_size, kernel_size=3, pool_size=2):
        """
        @params:
            - input_shape = shape of input data sample
            - panel_size = list of panel sizes or an int if only network depth=1
            - kernel_size = dimension of convolutional filter/kernel
            - pool_size = max-pool dimension for up/downsampling 
        """
        super().__init__()
        self.input_shape = input_shape
        self.panel_size = panel_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.depth = len(panel_size) \
              if isinstance(panel_size,list) else panel_size
        
        # encoder
        inputs = keras.Input(shape=self.input_shape, name='inputs')  # input image
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

        input_shape = self.encoder.input_shape
        print('shape of input image: ', input_shape[1:-1])
        print('shape of output image: ', decoded.shape[1:-1])

        decoded = layers.Conv2D(1, self.kernel_size, activation='sigmoid',
                                padding='same', name='outputs')(decoded)

        return decoded


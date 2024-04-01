from custom_classes_defs.setup import *

## ===================================================
class FNET2D(model_config):
    """
    Keras 2D image segmentation with a U-Net-like architecture;
    Taken from https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(self, panel_sizes, model_arch=None, **kwargs):
        """
        @params:
        img_shape : shape of image including panel dimension
        resnet50_path : path to pretrained model (resnet50)
        """
        if model_arch is None:
            super().__init__(**kwargs)
        else:
            super().__init__(**model_arch, **kwargs)

        self.Name = 'fnet'
        self.panel_sizes = panel_sizes

    def encoder_block(self, x, previous_block_activation, panel_sizes,block):

        for i, filters in enumerate(panel_sizes):
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            name = f'encoding_block_{i+1}{chr(block+65)}'
            x = layers.add([x, residual], name=name)  # Add back residual
            previous_block_activation = x  # Set aside next residual

        return x, previous_block_activation
    
    def decoder_block(self, x, previous_block_activation, panel_sizes, block):

        for i, filters in enumerate(panel_sizes[::-1]):
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)

            name = f'decoding_block_{i+1}{chr(block+65)}'
            x = layers.add([x, residual], name=name)  # Add back residual
            previous_block_activation = x  # Set aside next residual

        return x, previous_block_activation


    def compute_zero_padding(self):
        """ Assumes image shape is square;
        determines padding of initial image so that subsequent images 
        get halved each time an encoding step is applied """
        pad  = 0   
        L = len(self.panel_sizes)
        zero_paddding = True
        while zero_paddding:
            N = self.img_shape[0] + 2*pad 
            # infer all possible image sizes
            img_sizes = [N // pow(2, n) for n in range(L) if pow(2,n)<N]
            idx = np.mod(img_sizes[:-1], img_sizes[1:]).argmax()
            newL = len(img_sizes)
            if not(idx==0 and newL==L):
                pad = pad + 1
            else:
                zero_paddding = False
        return pad
    
    def build_model(self):
        inputs = keras.Input(shape=self.img_shape + (self.channels_dim[0],))

        pad = self.compute_zero_padding()
        inter_inputs = self.take_off(inputs, pad)

        # Entry block
        x = layers.Conv2D(self.panel_sizes[0], 3, strides=2, padding="same")(inter_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        d = 0
        N2 = len(self.panel_sizes)-2
        for i in range(len(self.panel_sizes)):
            self.entry_block = (i==0)
            e = 1 if i==0 else d
            d = max(N2-i,0)
            x, previous_block_activation = \
                self.encoder_block(
                    x, previous_block_activation, self.panel_sizes[e:], i
                )
            x, previous_block_activation = \
                self.decoder_block(
                    x, previous_block_activation, self.panel_sizes[d:], i
                )

        x = self.landing(x, pad)
        # Add a per-pixel classification layer
        activation="softmax" if self.channels_dim[1]>1 else "sigmoid"
        outputs = layers.Conv2D(
            self.channels_dim[1], 3, 
            activation=activation, 
            padding="same"
        )(x)

        return keras.Model(inputs, outputs)
    
    

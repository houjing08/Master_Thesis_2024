from custom_classes_defs.setup import *

## ===================================================
class FNET2D(model_config):
    """
    Keras 2D image segmentation with a U-Net-like architecture;
    Taken from https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(self, 
            panel_sizes, 
            model_arch=None, 
            drop_rate=0.1, 
            add_residual=True, 
            **kwargs
        ):
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
        self.depth = len(panel_sizes)
        self.drop_rate = drop_rate
        self.add_residual = add_residual

    def conv_block(self, num_filters, name):
        """ Double convolutions with batch-norms, activations and dropouts"""
        cblock = keras.Sequential(
            [
                layers.Conv2D(num_filters, 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.drop_rate),
                layers.Conv2D(num_filters, 3, padding="same", name=name),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.drop_rate)
            ]
        )
        return cblock
    
    # def deconv_block(self, num_filters):
    #     """ Double deconvolutions with activations and batch-norms"""
    #     dblock = keras.Sequential(
    #         [
    #             layers.Activation("relu"),
    #             layers.Conv2DTranspose(num_filters, 3, padding="same"),
    #             layers.BatchNormalization(),
    #             layers.Activation("relu"),
    #             layers.Conv2DTranspose(num_filters, 3, padding="same"),
    #             layers.BatchNormalization()
    #         ]
    #     )
    #     return dblock

    def encoder_blocks(self, x, encoder_blocks_outputs, panel_sizes,block):

        # encoder_blocks_outputs = [previous_block_activation]
        if not len(encoder_blocks_outputs):
            encoder_blocks_outputs = []

        depth = len(panel_sizes)
        for i, filters in enumerate(panel_sizes):
            name = f'encoder_block_{chr(block+65)}{i+1}'
            x = self.conv_block(filters, name=name)(x)

            if i < self.depth-1:
                if not block:
                    encoder_blocks_outputs.append(x)
                else:                                       # from intermediate encoders
                    encoder_blocks_outputs[i-depth] = x     # tentative

                x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        return x, encoder_blocks_outputs
    
    def decoder_blocks(self, x, encoder_blocks_outputs, panel_sizes, block):

        for i, filters in enumerate(panel_sizes[-2::-1]):

            if i < self.depth-1:    # top level
                x = layers.UpSampling2D(2)(x)
                x = layers.Conv2DTranspose(filters, 3, padding="same")(x)

            residual = encoder_blocks_outputs[-i-1]
            if residual.shape[-1] != filters:       # tentative
                residual = layers.Conv2D(filters, 1, padding="same")(residual)

            if self.add_residual:
                x = layers.add([x, residual])  # Add residual
            else:
                x = layers.Concatenate()([x, residual]) # concatenate

            name = f'decoder_block_{chr(block+65)}{i+1}'
            x = self.conv_block(filters,name)(x)

        return x

    def encoder_decoder(self, x):

        encoder_blocks_outputs = []
        for i in range(self.depth):
            d = min(i+2, self.depth)
            x, encoder_blocks_outputs = \
                self.encoder_blocks(
                    x, encoder_blocks_outputs, self.panel_sizes[-i:], i
                )
            x = self.decoder_blocks(
                    x, encoder_blocks_outputs, self.panel_sizes[-d:], i
                )  

        return x


    def compute_zero_padding(self):
        """ Assumes image shape is square;
        determines padding of initial image so that subsequent images 
        get halved each time an encoding step is applied """
        pad  = 0   
        L = self.depth
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
        x = inter_inputs

        x = self.encoder_decoder(x)

        x = self.landing(x, pad)
        # Add a per-pixel classification layer
        activation="softmax" if self.channels_dim[1]>1 else "sigmoid"
        outputs = layers.Conv2D(
            self.channels_dim[1], 3, 
            activation=activation, 
            padding="same"
        )(x)

        return keras.Model(inputs, outputs, name="F-NET")
    
    

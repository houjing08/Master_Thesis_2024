from custom_classes_defs.setup import *

## ===================================================
class UNET2D(model_config):
    """
    Keras 2D image segmentation with a U-Net-like architecture;
    Taken from https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(self, 
            panel_sizes, 
            model_arch=None, 
            drop_rate=0.25, 
            add_residual=True,
            **kwargs
        ):
        """
        @params:
        panel_sizes : list of filter sizes at each encoder layer
        model_arch : model architecture
        drop_rate : dropout rate (probability)
        """
        if model_arch is None:
            super().__init__(**kwargs)
        else:
            super().__init__(**model_arch, **kwargs)

        self.Name = 'unet'
        self.panel_sizes = panel_sizes
        self.drop_rate = drop_rate
        self.depth = len(panel_sizes)
        self.add_residual = add_residual


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
    
    def conv_block(self, num_filters):
        """ Double convolutions with batch-norms, activations and dropouts"""
        cblock = keras.Sequential(
            [
                layers.Conv2D(num_filters, 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.drop_rate),
                layers.Conv2D(num_filters, 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.drop_rate)
            ]
        )
        return cblock
    
    def deconv_block(self, num_filters):
        """ Double deconvolutions with activations and batch-norms"""
        dblock = keras.Sequential(
            [
                layers.Activation("relu"),
                layers.Conv2DTranspose(num_filters, 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2DTranspose(num_filters, 3, padding="same"),
                layers.BatchNormalization()
            ]
        )
        return dblock
            
    def build_model(self):
        inputs = keras.Input(shape=self.img_shape + (self.channels_dim[0],))

        pad = self.compute_zero_padding()
        inter_inputs = self.take_off(inputs, pad)
        x = inter_inputs

        ### Encoder ###
        encode_outputs = []
        for i, filters in enumerate(self.panel_sizes):
            x = self.conv_block(filters)(x)

            if i < self.depth-1:
                encode_outputs.append(x)
                x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        ### Decoder ###
        for i, filters in enumerate(self.panel_sizes[-2::-1]):
            if i < self.depth-1:
                x = layers.UpSampling2D(2)(x)
                x = layers.Conv2DTranspose(filters, 3, padding="same")(x)

            if self.add_residual:
                x = layers.add([x, encode_outputs[-i-1]])  # Add residual
            else:
                x = layers.Concatenate()([x, encode_outputs[-i-1]]) # concatenate

            x = self.conv_block(filters)(x)

        x = self.landing(x, pad)
        # Add a per-pixel classification layer
        activation="softmax" if self.channels_dim[1]>1 else "sigmoid"
        outputs = layers.Conv2D(
            self.channels_dim[1], 3, 
            activation=activation, 
            padding="same"
        )(x)

        return keras.Model(inputs, outputs, name="U-NET")
    
    

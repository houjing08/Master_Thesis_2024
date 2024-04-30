from custom_classes_defs.setup import *

import re

## ===================================================
class HED2D(model_config):
    """
    Holistic Edge Detection (2D neural network for semamtic edge detection)
    Cf. Xie & Tu (2015)
    Uses VGG16 as backbone network
    """
    def __init__(self,  vgg16_path=None, num_freeze=4, model_arch=None, **kwargs):
        """
        @params:
        img_shape : shape of image including panel dimension
        vgg16_path : path to pretrained model (vgg16)
        num_freeze : number of convolutional base blocks to freeze (0-5)
        """
        if model_arch is None:
            super().__init__(**kwargs)
        else:
            super().__init__(**model_arch, **kwargs)

        self.vgg16_path = vgg16_path
        self.num_freeze =  min(5, max(0, int(num_freeze)))
        self.Name = 'hed'


    def compute_zero_padding(self):
        """ Assumes image shape is square;
        determines padding of initial image so that subsequent images 
        get halved each time an encoding step is applied """
        pad  = 0   
        L = 5
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

        preprocessed = keras.applications.vgg16.preprocess_input(inter_inputs)

        vgg16 = keras.applications.VGG16(
            include_top=False,
            weights="imagenet" if (self.vgg16_path is None) 
                                else self.vgg16_path, 
            input_tensor=preprocessed
        )

        conv_blocks = []
        for i, name in enumerate([
            'block1_conv2','block2_conv2','block3_conv3','block4_conv3','block5_conv3'
            ]):
            x = vgg16.get_layer(name=name).output
            x = layers.Conv2D(1, 1, padding="same")(x)
            if i>1:
                x = layers.Conv2DTranspose(1, 1, strides=2**i, padding="same")(x)
            x = self.landing(x, pad)
            conv_blocks.append(x)

        x = layers.concatenate(conv_blocks)
        x = layers.Conv2D(1, 1, padding="same")(x)
        x = self.landing(x, pad)
        conv_blocks.append(x)

        # Add a per-pixel classification layer
        outputs = [
            layers.Conv2D(
            self.channels_dim[1], 3, 
            activation="sigmoid", 
            padding="same")(x) \
            for x in conv_blocks
        ]

        model =  keras.Model(inputs, outputs, name="HED")

        # Enforce transfer learning
        for ly in model.layers:
            if re.search(f'block[0-{self.num_freeze}]', ly.name):
                ly.trainable = False

        return model
    

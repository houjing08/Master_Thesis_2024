from custom_classes_defs.setup import *

## ===================================================
class DeeplabV3Plus(model_config):
    """
    Keras multiclass semantic segmentation using DeepLabV3+;
    Taken from https://keras.io/examples/vision/deeplabv3_plus/
    """
    def __init__(self, resnet50_path=None, model_arch=None, **kwargs):
        """
        @params:
        img_shape : shape of image including panel dimension
        resnet50_path : path to pretrained model (resnet50)
        """
        if model_arch is None:
            super().__init__(**kwargs)
        else:
            super().__init__(**model_arch, **kwargs)
            
        self.resnet50_path = resnet50_path
        self.update_model_arch(model_arch)
        self.Name = 'DeepLabV3+'

    def convolution_block(self,
            block_input,
            num_filters=256,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
        ):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
        )(block_input)
        x = layers.BatchNormalization()(x)
#         return ops.nn.relu(x)
        return layers.Activation('relu')(x)


    def DilatedSpatialPyramidPooling(self, dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = self.convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear",
        )(x)
    
        out_1 = self.convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    
        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolution_block(x, kernel_size=1)
        return output


    def pad_if_not_size_power_2(self, n):
        if (n!=0) and (n & (n-1) == 0):
            return n, 0
        k = 2
        while k < n and k < 100000:
            k = k * 2
        return k, (k-n)//2
    
    def build_model(self):
        model_input = keras.Input(shape=self.img_shape+(self.channels_dim[0],))
        image_size, _ = self.img_shape
        image_size, pad = self.pad_if_not_size_power_2(image_size)
    
        intermediate = False
        if (pad != 0 and self.channels_dim[0] == 1):
            padded = layers.ZeroPadding2D(padding=pad)(model_input)
            inter_input = layers.Concatenate(axis=-1)([padded]*3)
            intermediate = True
        elif (pad != 0):
            inter_input = layers.ZeroPadding2D(padding=pad)(model_input)
            intermediate = True
        elif self.channels_dim[0] == 1:
            inter_input = layers.Concatenate(axis=-1)([model_input]*3)
            intermediate = True

        inter_input = self.take_off(inter_input if intermediate else model_input)

        preprocessed = keras.applications.resnet50.preprocess_input(
                   inter_input
                )
    
        resnet50 = keras.applications.ResNet50(
            weights="imagenet" if (self.resnet50_path is None) 
                                else self.resnet50_path, 
            include_top=False,
            input_tensor=preprocessed
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.DilatedSpatialPyramidPooling(x)
    
        input_a = layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)
    
        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)
        x = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        x = self.landing(x)
        model_output = layers.Conv2D(
            self.channels_dim[1], kernel_size=(1, 1), padding="same",
            activation="softmax" if self.channels_dim[1]>1 else "sigmoid", 
        )(x)

        return keras.Model(inputs=model_input, outputs=model_output)
    
    

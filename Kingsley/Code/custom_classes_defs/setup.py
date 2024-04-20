import tensorflow as tf
import keras
from keras import layers

import numpy as np
import pandas as pd
import time
# import json
import pickle
# import cv2
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, average_precision_score

class model_config(keras.Model):
    """ Define common parameters required for training any model """
    def __init__(
            self, 
            optimizer='adam', 
            loss='binary_crossentropy', 
            epochs=1,
            batch_size=1, 
            shuffle=True, 
            verbose=0,
            new_training_session=False,
            save_path='',
            img_shape=(28,28), 
            target_size=(28,28), 
            channels_dim=(1,1),     # i/o
            augmentation=False,
            scaling=255.,
            threshold=0.5,
            pos_label=1,
            mixed_precision=None,
            **kwargs
        ):
        super().__init__(**kwargs)

        self.compile_args = dict(optimizer=optimizer, loss=loss)
        self.training_args = dict(epochs=epochs, batch_size=batch_size,
                                  shuffle=shuffle, verbose=verbose)
        self.new_training_session = new_training_session
        self.save_path = save_path
        self.threshold = threshold
        self.pos_label = pos_label
        self.mixed_precision = mixed_precision

        if mixed_precision:
            # Necessary to speed up computations on GPU
            if isinstance(mixed_precision, str):
                keras.mixed_precision.set_global_policy(mixed_precision)
            else:
                keras.mixed_precision.set_global_policy("mixed_float16")

        self.model_arch=dict(
            img_shape=img_shape, 
            target_size=target_size, 
            channels_dim=channels_dim,
            scaling=scaling,
            augmentation=augmentation
        )
        self.update_model_arch(self.model_arch)
    
    def set(self, conf_type='training', **kwargs):
        """ Resets default model compile and fit parameters """
        if conf_type=='compile':
            for key,value in kwargs.items():
                self.compile_args[key] = value
        else: # 'train'
            for key,value in kwargs.items():
                self.training_args[key] = value


    def update_model_arch(self, model_arch):
        """ Update model architecture parameters """
        if isinstance(model_arch, dict):
            try:
                self.img_shape = model_arch['img_shape']
                self.target_size = model_arch['target_size']
                self.channels_dim = model_arch['channels_dim']
                self.scaling = model_arch['scaling']
                self.augmentation = model_arch['augmentation']
            except KeyError:
                print(f'Unown key passed for model_arch!')

    
    def info(self, prettyprint=True):
        text = []
        for param, value in [
            ('compile_args', self.compile_args),
            ('training_args', self.training_args),
            ('model_arch', self.model_arch),
            ('new_training_session', self.new_training_session),
            ('save_path', self.save_path),
            ('threshold', self.threshold),
            ('pos_label', self.pos_label)
        ]:
            if isinstance(value, dict):
                text.append("{:>20}:".format(param))
                for k, v in value.items():
                    if isinstance(v, (str,bool,int,float,tuple)):
                        text.append("{:>30}: {}".format(k, v))
                    elif isinstance(v, list):
                        l = min(len(v),10)
                        text.append("{:>30}: {}".format(k, v[0]))
                        for val in v[1:l]:
                            text.append("{:>30}: {}".format('', val))
                    else:
                        text.append("{:>30}: {}".format(k, type(v)))
            else:
                text.append("{:>20}: {}".format(param, value))
        text = '\n'.join(text)
        if prettyprint:
            print(text)
        else:
            return text
        
    def double_check(self, interactive=True):
        """ Cross-examine some setup parameters """
        if interactive:
            train = input("New train session? (y/n): ")
            if train[0].lower()=='y':
                self.new_training_session = True
            else:
                self.new_training_session = False

    def execute_training(self, model, data, saveas='model',
                          metrics=['loss','val_loss'], plot_history=True ):
        if self.new_training_session:
            model.compile(**self.compile_args)
            print('Model training...')
            start = time.time()
            history = model.fit(data, **self.training_args)
            print('training elapsed time: ___{:5.2f} minutes___'.format((time.time()-start) / 60))
            print('...training completed!')
            if plot_history:
                show_convergence(history.history, metrics=metrics)
            model.save_weights(join(self.save_path, saveas+'.weights.h5'))
            with open(join(self.save_path, saveas+'_info.txt'), "w") as fp:
                fp.write(self.info(0))
            try:
                with open(join(self.save_path, saveas+'_history.pickle'), "wb") as fp:
                    pickle.dump(history, fp)
            except TypeError:
                print(f"Error writing {saveas+'_history.pickle'}!")
        else:
            try:
                model.load_weights(join(self.save_path, saveas+'.weights.h5'))
                with open(join(self.save_path, saveas+'_history.pickle'), "rb") as fp:
                    history = pickle.load(fp)
                print(f"model weights  ({saveas}.weights.h5) and train history loaded!")
                print(open(join(self.save_path, saveas+'_info.txt'),'r').read())
                if plot_history:
                    show_convergence(history.history, metrics=metrics)
            except FileNotFoundError:
                self.new_training_session = True
                self.execute_training(model, data, saveas)
                self.new_training_session = False
        return history


    # def predict(self, x_test, source='images', target='images'):

    #     if (source=='codes' and target=='images'):
    #         return self.decoder.predict(x_test)

    #     elif (source=='images' and target=='codes'):
    #         return self.encoder.predict(x_test)

    #     return self.autoencoder.predict(x_test) 

    def evaluate_sklearn(self, y_true, y_pred, report=False):
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
        eval_time = time.time()
        neg_label = int(not(self.pos_label))
        y_true = y_true.flatten()
        AP = average_precision_score(y_true, y_pred.flatten(), pos_label=self.pos_label)

        # ODS
        y_true = y_true.astype(bool)
        labels = [bool(neg_label), bool(self.pos_label)]

        if self.pos_label:
            cut = np.quantile(y_pred, q=self.threshold)
            y = (y_pred > cut).flatten()
        else:
            cut = np.quantile(y_pred, q=1-self.threshold)
            y = (y_pred < cut).flatten()

        results = classification_report(y_true, y, output_dict=True, labels=labels)
        f1_ods = results[str(labels[-1])]['f1-score']

        #OIS
        if self.pos_label:
            cuts = np.quantile(y_pred, q=self.threshold, axis=(1,2))
            cuts = np.expand_dims(np.expand_dims(cuts,-1),-1)
            y = (y_pred > cuts).flatten()
        else:
            cuts = np.quantile(y_pred, q=1-self.threshold, axis=(1,2))
            cuts = np.expand_dims(np.expand_dims(cuts,-1),-1)
            y = (y_pred < cuts).flatten()

        results = classification_report(y_true, y, output_dict=True, labels=labels)
        f1_ois = results[str(labels[-1])]['f1-score'] 

        eval_time = time.time() - eval_time
        print("evaluation elapsed time:___{:5.2f}___minutes" \
                .format(eval_time/60))
        
        if report:
            pd.DataFrame(results).round(2).style

        return {'Avg-precision': np.round(AP,2), 'f1-score(ODS)':  np.round(f1_ods,2), 'f1-score(OIS)': np.round(f1_ois,2)}


    def equal(self, shape1, shape2):
        if len(shape1) != len(shape2):
            return False
        return (np.array(shape1 == shape2)).all()

    def take_off(self, inputs, pad=0):
        """ - Checks compatibility of input layer with input image shape    
            - Applies data augmentation if requested.  """
        if self.augmentation:
            inputs = self.add_augmentation(inputs)

        if self.scaling > 1:
            inputs = layers.Rescaling(scale=1./self.scaling)(inputs)

        # if not self.equal(self.img_shape, self.target_size):
        #     inputs = layers.Resizing(
        #             width=self.target_size[0], height=self.target_size[1],
        #             crop_to_aspect_ratio=True
        #         )(inputs)

        if (pad):
            inputs = layers.ZeroPadding2D(padding=pad)(inputs)
            print(f"inputs padded by {pad} to fit model design")

        return inputs

    def landing(self, outputs, pad=0):
        """ Checks compatibility of output layer with input image shape """
        if pad>0:
            print(f'Cropping the output by {pad} to fit input...')
            outputs = layers.Cropping2D(pad)(outputs)

        if not self.equal(outputs.shape[1:-1], self.target_size):
            print('Resizing the output to fit input...')
            outputs = layers.Resizing(
                    width=self.img_shape[0], height=self.img_shape[1],
                    crop_to_aspect_ratio=True
                )(outputs)
        return outputs

    def add_augmentation(self, inputs):
        """ adds a few transformations for intrinsic data augmentation """
        data_aug = keras.Sequential(
                [
                    layers.RandomFlip('horizontal'),
                    layers.RandomContrast(factor=0.2),
                    layers.RandomRotation(factor=0.2), # value_range=(0,self.scaling)),
                    layers.RandomBrightness(factor=0.2), # value_range=(0,self.scaling)),
                    layers.RandomFlip('vertical')
                ]
            )
        return data_aug(inputs)






## ===================================================
def display_sample_images(x_test, y, img_shape, n=10, figsize=(20,4), cmap=None):
    n = 10  # How many digits we will display
    plt.figure(figsize=figsize)
    for k, i in enumerate(np.random.randint(0,x_test.shape[0],size=n)):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        ax.imshow(x_test[i].reshape(img_shape), cmap=cmap)
        ax.set_axis_off()
    
        # Display reconstruction
        ax = plt.subplot(2, n, k + 1 + n)
        ax.imshow(y[i].reshape(img_shape), cmap=cmap)
        ax.set_axis_off()
    plt.show()

## ===================================================
def show_convergence(history, metrics='loss'):
    """ Plot model train/validation loss history
    in order to monitor convergence over epochs """
    if isinstance(metrics, list):
        for metric in metrics:
            if metric in history:
                plt.plot(history[metric], label=metric)
            else:
                print(f'cannot find {metric} in history')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(metrics[0])
        plt.show()
    elif metrics in history:
        plt.plot(history[metrics])
        plt.xlabel('epoch')
        plt.ylabel(metrics)
        plt.show()
    else:
        print(f'cannot find {metrics} in history')


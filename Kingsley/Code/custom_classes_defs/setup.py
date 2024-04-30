import tensorflow as tf
import keras
from keras import layers
import keras.backend as KB

import numpy as np
# from IPython.display import display
from glob import glob
import pandas as pd
import time
# import json
import pickle
# import cv2
import os
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
            shuffle=False, 
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
            multiple_gpu_device=None,
            training_duration=None,
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
        self.labels = list(range(max(channels_dim[1],2)))
        self.mixed_precision = mixed_precision
        self.multiple_gpu_device = multiple_gpu_device
        self.training_duration = training_duration

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
            ('pos_label', self.pos_label),
            ('labels', self.labels),
            ('mixed_precision', self.mixed_precision),
            ('multiple_gpu_device', self.multiple_gpu_device),
            ('training_duration (mins)', self.training_duration)
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
                          metrics=['loss','val_loss'], plot_history=True):
        
        if self.new_training_session:
            for chkpt in glob(self.save_path+'/*.h5'):
                os.remove(chkpt)
            print('Model training...')
            start = time.time()
            history = model.fit(data, **self.training_args)
            self.training_duration = f'{(time.time()-start) / 60:.2f}'
            print('training elapsed time: ___{:5}___ minutes'.format(self.training_duration))
            print('...training completed!')

            if plot_history:
                show_convergence(history.history, metrics=metrics)
            with open(join(self.save_path, saveas+'_info.txt'), "w") as fp:
                fp.write(self.info(0))
            try:
                with open(join(self.save_path, saveas+'_history.pickle'), "wb") as fp:
                    pickle.dump(history, fp)
            except TypeError:
                print(f"Error writing {saveas+'_history.pickle'}!")

        else:

            try:
                with open(join(self.save_path, saveas+'_history.pickle'), "rb") as fp:
                    history = pickle.load(fp)
                print(f"model train history '{saveas}+_history.pickle'loaded!")
                print(open(join(self.save_path, saveas+'_info.txt'),'r').read())
                if plot_history:
                    show_convergence(history.history, metrics=metrics)
            except FileNotFoundError:
                print('No saved model/weights found!')
                model, history = None, None

        best_model_track = sorted(glob(self.save_path+'/*.h5'))
        if len(best_model_track):
            mode = best_model_track[0].__contains__('loss')-1
            model.load_weights(best_model_track[mode])

        return model, history

    def callbacks(self,
        es_monitor='accuracy', 
        lr_monitor='val_accuracy', 
        chkpt_monitor='val_loss',
        es_patience=10,
        lr_patience=10
    ):
        """ Define of callback methods to use for training (some params hard-coded!)"""
        earlystopping = keras.callbacks.EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            restore_best_weights=True
        )

        reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
            monitor=lr_monitor, 
            factor=0.5, 
            patience=lr_patience, 
            min_lr=1e-6,
            verbose=1
        )
        filename =  "chkpt-"+chkpt_monitor+"-{val_loss:.4f}-{epoch:02d}.weights.h5"\
                    if chkpt_monitor.__contains__('loss') else \
                     "chkpt-"+chkpt_monitor+"-{val_accuracy:.4f}-{epoch:02d}.weights.h5"

        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(self.save_path, filename), 
            monitor=chkpt_monitor,
            save_weights_only=True,
            save_best_only=True
        )

        schedule_lr = keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * KB.exp(-0.1) if (epoch > 10) else lr,
            verbose=1
        )

        callback_list = [
            reduce_lr_on_plateau,
            checkpoint,
            # schedule_lr,
            earlystopping
        ]
        return callback_list

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
        if hasattr(y_true, 'as_numpy_iterator'): # tensorflow dataset
            y_true = np.concatenate([a[-1] for a in y_true.as_numpy_iterator()])
        y_true = y_true.flatten()
        y = y_pred.reshape(-1, self.channels_dim[1])
        AP = average_precision_score(y_true, y, pos_label=self.pos_label)

        # ODS (single threshold for all images)
        if self.pos_label and self.channels_dim[1]==1:
            cut = np.quantile(y_pred, q=self.threshold)
            y = (y_pred > cut).astype('uint8').flatten()
        elif self.channels_dim[1]==1:
            cut = np.quantile(y_pred, q=1-self.threshold)
            y = (y_pred < cut).astype('uint8').flatten()
        else:
            y = np.argmax(y_pred, axis=-1).astype('uint8').flatten()

        results = classification_report(y_true, y, output_dict=True, labels=self.labels)
        f1_ods = results[str(self.pos_label)]['f1-score']

        eval_time = time.time() - eval_time
        print("evaluation elapsed time:___{:5.2f}___minutes" \
                .format(eval_time/60))
        
        if report:
            print(pd.DataFrame(results).round(2))

        return {'Avg-precision': np.round(AP,2), 'f1-score(ODS)':  np.round(f1_ods,2)}


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

# #################################################################
#         # from keras.io
# #################################################################
       
# class MyTrainer(keras.Model):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         # Create loss and metrics here.
#         self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
#         self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy()

#     @property
#     def metrics(self):
#         # List metrics here.
#         return [self.accuracy_metric]

#     def train_step(self, data):
#         x, y = data
#         with tf.GradientTape() as tape:
#             y_pred = self.model(x, training=True)  # Forward pass
#             # Compute loss value
#             loss = self.loss_fn(y, y_pred)

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         # Update metrics
#         for metric in self.metrics:
#             metric.update_state(y, y_pred)

#         # Return a dict mapping metric names to current value.
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
#         x, y = data

#         # Inference step
#         y_pred = self.model(x, training=False)

#         # Update metrics
#         for metric in self.metrics:
#             metric.update_state(y, y_pred)
#         return {m.name: m.result() for m in self.metrics}

#     def call(self, x):
#         # Equivalent to `call()` of the wrapped keras.Model
#         x = self.model(x)
#         return x

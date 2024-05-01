import struct, os
import random
import time
import cv2
from array import array
from os.path import join
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# from tensorflow import ops

from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


# from IPython.display import Image, display
from keras.utils import load_img, array_to_img , img_to_array
from PIL import ImageOps

## ==================================================
class Preprocess():
    """ Common preprocessing tools """
    def __init__(self, threshold=0.5, pos_label=1):
        self.scaler = MinMaxScaler()   
        self.threshold = threshold
        self.pos_label = pos_label

    def transform(self, X):
        return self.scaler.transform(X.reshape(-1,1)).reshape(X.shape)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X.reshape(-1,1)).reshape(X.shape)

    def normalize(self, data, add_panel=True):
        if add_panel:
            data = np.expand_dims(data, -1)
        return self.transform(data.astype("float32"))   

    # def get_labels(self, X, normalize=False):
    #     """ Determine labels on a matrix of probabilities p \in [0,1]:
    #         label = 1 if p > threshold else 0. """
    #     if normalize:
    #         X = self.normalize(X)
    #     if self.pos_label:
    #         # cut = np.quantile(X, q=self.threshold)
    #         cuts = np.quantile(X, q=self.threshold, axis=(1,2))
    #         cuts = np.expand_dims(np.expand_dims(cuts,-1),-1)
    #         return (X > cuts).astype('int')
    #     else:
    #         # cut = np.quantile(X, q=1-self.threshold)
    #         cuts = np.quantile(X, q=1-self.threshold, axis=(1,2))
    #         cuts = np.expand_dims(np.expand_dims(cuts,-1),-1)
    #         return (X < cuts).astype('int')
        
    def get_labels(self, X, normalize=False):
        """ Determine labels on an image with intensities p \in [0,255]:
            using OTSU's thresholding """
        labels = self.Otsu_thresholding(X)

        if self.pos_label:
            return (labels > 0).astype('int')
        else:
            return (labels <= 0).astype('int')
        
    # def get_labels(self, X, normalize=False):
    #     """ Determine labels on an image with intensities p \in [0,255]:
    #         using OTSU's thresholding """

    #     if self.pos_label:
    #         return (X > self.threshold).astype('int')
    #     else:
    #         return (X < self.threshold).astype('int')


    def data_generator(self, data, batch_size=1, normalize=True, as_numpy=False):
        X, y =  self.normalize(data) if normalize else data, \
                self.get_labels(data,normalize)
        if not(as_numpy):
            return X, y
        dataset = tf_data.Dataset.from_tensor_slices((X,y)).batch(batch_size)
        return dataset
    
    def Otsu_thresholding(self, X):
        img = X.astype('uint8')
        labels = []
        for i in range(len(X)):
            _, label = cv2.threshold(img[i], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            labels.append(label)
        # blur = cv2.GaussianBlur(img[i],(5,5),0)
        # _, label = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return np.array(labels)


## ==================================================
#
# MNIST Data Loader Class: From Keras-io
#
class MNIST_digits(Preprocess):
    def __init__(self, input_path='', 
                 training_images_filepath='',
                 training_labels_filepath='', 
                 test_images_filepath='', 
                 test_labels_filepath='',
                 threshold=0.5,
                 pos_label=1
                 ):
        super().__init__(threshold=threshold, pos_label=pos_label)

        def join(string1, string2):
            return os.path.join(input_path,string1) \
                    if len(string1) else os.path.join(input_path, string2)

        self.training_images_filepath = join(training_images_filepath, 
                                             'train-images-idx3-ubyte/train-images-idx3-ubyte')
        self.training_labels_filepath = join(training_labels_filepath, 
                                             'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        self.test_images_filepath = join(test_images_filepath, 
                                         't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        self.test_labels_filepath = join(test_labels_filepath, 
                                         't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath,self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )

        min_val = np.min(x_train)
        max_val = np.max(x_train)
        self.scaler = self.scaler.fit([[min_val],[max_val]])

        # img = np.concatenate([x_train, x_test]).astype('uint8')
        # # blur = cv2.GaussianBlur(img,(5,5),0)
        # self.threshold, _ = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return (np.array(x_train), np.array(y_train)),(np.array(x_test), np.array(y_test))    


## ==================================================
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

## ==================================================
class MNIST_letters(Preprocess):
    """ Handles loading, scaling and preprocessing of MNIST handwritten letters """
    def __init__(self, url,threshold=0.5, pos_label=1,edge_letters=True):
                                 
        super().__init__(threshold=threshold, pos_label=pos_label)
        self.threshold = threshold
        self.pos_label = pos_label

        # Load Hand-written alphabet images for input url (path)
        self.df = pd.read_csv(url)
        self.df.set_index('0', inplace=True)
        self.labels = dict(zip(range(0,26),map(chr, range(ord('A'), ord('Z')+1))))

        # Select letters with straight edges
        if edge_letters:
            self.df = self.df.loc[[ord(s)-ord(self.labels[0]) for s in list('AEFHIKLMNTVWXYZ')]]

        # defiine min_max_scaler normalization and fit on data
        min_val = self.df.iloc[:,1:].min(axis=1).min()
        max_val = self.df.iloc[:,1:].max(axis=1).max()
        self.scaler = self.scaler.fit([[min_val],[max_val]])

        # img = self.df.values.astype('uint8')
        # # blur = cv2.GaussianBlur(img,(5,5),0)
        # threshold, _ = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    def split_data(self, train_ratio = 0.1, img_dim=28, val_ratio = 0.0):
        """ Split data into train-test proportions, given train-ratio, and re-arrange"""
        x_train, x_test, y_train, y_test = \
        train_test_split(self.df.values, self.df.index.values, train_size=train_ratio)
        x_train = np.array([x.reshape(img_dim,img_dim) for x in x_train])
        x_test = np.array([x.reshape(img_dim,img_dim) for x in x_test])

        if abs(val_ratio) > 0:
            # take fraction of remaining data for validation
            test_ratio = val_ratio / (1-train_ratio)
            x_test, x_val, y_test, y_val = \
            train_test_split(x_test, y_test, test_size=test_ratio)
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)

        return (x_train, y_train), (x_test, y_test)

## ==================================================
class Oxford_Pets(Preprocess):
    """
    Loading, scaling and preprocessing of the Oxford pets data:
    Returns: tf-datasets
    Ref: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(
            self, 
            input_dir,
            target_dir,
            img_size=(160, 160),
            threshold=0.5, 
            pos_label=1,
            batch_size=1
        ):
        super().__init__(threshold=threshold, pos_label=pos_label)

        self.img_size = img_size
        self.threshold = threshold
        self.pos_label = pos_label
        self.batch_size = batch_size

        self.input_img_paths = sorted(
            [
                join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".jpg")
            ]
        )
        self.target_img_paths = sorted(
            [
                join(target_dir, fname)
                for fname in os.listdir(target_dir)
                if fname.endswith(".png") and not fname.startswith(".")
            ]
        )
        print("Number of samples:", len(self.input_img_paths))

    def load_img_masks(self, input_img_paths, target_img_paths):
        input_img = tf_io.read_file(input_img_paths)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, self.img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_paths)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, self.img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        target_img -= 1
        return input_img, target_img

    def data_generator(self, input_img_paths, target_img_paths, cache=False):
        """Returns a TF Dataset.
        Cache data to memory depending on available GPU memory.
        """

        dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
        dataset = dataset.map(self.load_img_masks, num_parallel_calls=tf_data.AUTOTUNE) \
                 .batch(self.batch_size)
        return dataset.cache() if cache else dataset
    
    def split_data(self, train_ratio = 0.1, val_ratio = 0.1, seed=None, cache=False):
        """ 
        Extracts and splits data into training and validation sets 
        train_ratio = fraction of entire data
        val_ratio = fraction of entire data
        """
        start = time.time()
        L = len(self.input_img_paths)
        train_samples = int(train_ratio*L)
        val_samples = int(val_ratio*L)

        random.Random(seed).shuffle(self.input_img_paths)
        random.Random(seed).shuffle(self.target_img_paths)
        self.train_input_paths = self.input_img_paths[:train_samples]
        self.train_target_paths = self.target_img_paths[:train_samples]
        self.val_input_paths = self.input_img_paths[train_samples:][:val_samples]
        self.val_target_paths = self.target_img_paths[train_samples:][:val_samples]
        self.test_input_paths = self.input_img_paths[train_samples:][val_samples:]
        self.test_target_paths = self.target_img_paths[train_samples:][val_samples:]

        # Instantiate dataset for each split
        train_dataset = self.data_generator(self.train_input_paths, self.train_target_paths, cache)
        valid_dataset = self.data_generator(self.val_input_paths, self.val_target_paths, cache)
        test_dataset = self.data_generator(self.test_input_paths, self.test_target_paths)
        print(f'Data prep. duration: ___{(time.time()-start)/60:.2f}___ minutes.')

        return train_dataset, valid_dataset, test_dataset
    
    def plot_multiple(self, img_path, idx, rows, num_images, count, mask=False):
        for col in range(num_images):
            plt.subplot(rows, num_images, count)
            plt.axis('off')
            if isinstance(img_path, list):
                img = ImageOps.autocontrast(load_img(img_path[idx[col]])) \
                            if mask else \
                        cv2.imread(filename=img_path[idx[col]])
                # img = cv2.imread(filename=img_path[idx[col]])
                img = tf_image.resize(img, self.img_size, method="nearest")
                plt.imshow(img)
            else: # numpy array (img_path=y_pred)
                mask = np.argmax(img_path[idx[col]], axis=-1)
                mask = np.expand_dims(mask, axis=-1) * 127.5
                img = ImageOps.autocontrast(array_to_img(mask))
                #display(img)
                plt.imshow(img,cmap='gray')
            count = count + 1
        return count

    def display_sample_image(self, y_pred, validation='val'):
        """Quick utility to display a model's prediction."""
        num_images = 4
        idx = np.random.choice(np.arange(len(y_pred)), num_images)
        if validation.lower()=='val':
            img_path = self.val_input_paths
            target_path = self.val_target_paths
        elif validation.lower()=='test':
            img_path = self.test_input_paths
            target_path = self.test_target_paths
        else:
            img_path = self.train_input_paths
            target_path = self.train_target_paths

        plt.figure(figsize=(30,20)) 
        rows = 3
        count = 1
        # Display input image
        count = self.plot_multiple(img_path, idx, rows, num_images, count)

        # Display ground-truth target mask
        count = self.plot_multiple(target_path, idx, rows, num_images, count, mask=True)

        # Display mask predicted by our model
        count = self.plot_multiple(y_pred, idx, rows, num_images, count)
        plt.subplots_adjust(hspace=0)


## ==================================================
class Thebe(Preprocess):
    """ Handles loading of Thebe seismic data and labels into tf.datasets:
        training, validation and test datasets.
        Data stored as numpy array in npy format in some url (path).
      """
    def __init__(self, 
                seismic_url, annotations_url, batch_size,
                threshold=0.5, pos_label=1, seed=None
        ):
                                 
        super().__init__(threshold=threshold, pos_label=pos_label)
        self.threshold = threshold
        self.pos_label = pos_label
        self.seismic_url = seismic_url
        self.annotations_url = annotations_url
        self.seed = seed
        self.batch_size = batch_size
        self.val_target_paths = glob(join(annotations_url,'val/*.npy'))
        self.test_target_paths = glob(join(annotations_url,'test/*.npy'))
        self.train_target_paths = glob(join(annotations_url,'train/*.npy'))
        self.val_input_paths = glob(join(seismic_url,'val/*.npy'))
        self.test_input_paths = glob(join(seismic_url,'test/*.npy')) 
        self.train_input_paths = glob(join(seismic_url,'train/*.npy'))


    def load_img_masks(self, imgs_paths, labels_paths):
        imgs = np.stack(
            [np.load(img).astype('float32') for img in imgs_paths]
        )
        labels = np.stack(
            [np.load(label).astype('float32') for label in labels_paths]
        )
        return imgs, labels

    
    def data_generator(self, sub_group='train', as_numpy=False):
        """ 
        Extracts and splits data into training and validation sets 
        train_ratio = fraction of entire data
        val_ratio = fraction of entire data
        """
        path_to_imgs = join(self.seismic_url, sub_group)
        path_to_labels = join(self.annotations_url, sub_group)
        imgs_paths = sorted(glob(path_to_imgs+'/*'))
        labels_paths = sorted(glob(path_to_labels+'/*'))

        random.Random(self.seed).shuffle(imgs_paths)
        random.Random(self.seed).shuffle(labels_paths)

        if as_numpy:
            return self.load_img_masks(imgs_paths, labels_paths)
        dataset = tf_data.Dataset.from_tensor_slices(self.load_img_masks(imgs_paths, labels_paths))

        return dataset.batch(self.batch_size)
    
    def plot_multiple(self, img_path, idx, rows, num_images, count):
        for col in range(num_images):
            plt.subplot(rows, num_images, count)
            plt.axis('off')
            if isinstance(img_path, list):
                img = np.load(img_path[idx[col]])
                plt.imshow(img)
                #img = tf_image.resize(img, self.img_size, method="nearest")
                plt.imshow(img)
                
            else: # numpy array (img_path=y_pred)
            
                mask = img_path[idx[col]]
                img = ImageOps.autocontrast(array_to_img(mask))
                plt.imshow(img,cmap='gray')
            count = count + 1 
            
        return count
    
    
    def display_sample_image(self, y_pred, validation='val'):
        """Quick utility to display a model's prediction."""
        num_images = 8
        idx = np.random.choice(np.arange(len(y_pred)), num_images)
        if validation.lower()=='val':
            img_path = self.val_input_paths
            target_path = self.val_target_paths
        elif validation.lower()=='test':
            img_path = self.test_input_paths
            target_path = self.test_target_paths
        else:
            img_path = self.train_input_paths
            target_path = self.train_target_paths

        plt.figure(figsize=(30,20)) 
        rows = 3
        count = 1
        # Display input image
        count = self.plot_multiple(img_path, idx, rows, num_images, count)
        

        # Display ground-truth target mask
        count = self.plot_multiple(target_path, idx, rows, num_images, count)

        # Display mask predicted by our model
        count = self.plot_multiple(y_pred, idx, rows, num_images, count)
        plt.subplots_adjust(hspace=0)
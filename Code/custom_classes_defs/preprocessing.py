import struct, os
import random
import pickle
import re
import time
import cv2
from array import array
from os import walk
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import segyio
from typing import Tuple
from itertools import compress,product
from skimage.util.shape import view_as_windows
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
        self.train_size = None
        self.test_size = None
        self.validation_size = None

    def transform(self, X):
        return self.scaler.transform(X.reshape(-1,1)).reshape(X.shape)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X.reshape(-1,1)).reshape(X.shape)

    def normalize(self, data, add_panel=True):
        if add_panel:
            data = np.expand_dims(data, -1)
        return self.transform(data.astype("float32"))   
        
    def get_labels(self, X):
        """ Determine labels on an image with intensities p \in [0,1] """

        if self.pos_label:
            return (X > self.threshold).astype('uint8')
        else:
            return (X < self.threshold).astype('uint8')


    def data_generator(self, data, batch_size=1, normalize=True, as_numpy=False, cache=False):
        X = self.normalize(data) if normalize else data
        y = self.get_labels(X)

        if as_numpy:
            return X, y
        
        dataset = tf_data.Dataset.from_tensor_slices((X,y)).batch(batch_size)
        return dataset.cache() if cache else dataset


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
    def __init__(self, url, threshold=0.5, pos_label=1, edge_letters=True):
                                 
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

    def split_data(self, train_ratio = 0.1, img_dim=28, val_ratio = 0.0, random_state=None):
        """ Split data into train-test proportions, given train-ratio, and re-arrange"""
        x_train, x_test, y_train, y_test = \
        train_test_split(self.df.values, self.df.index.values, 
                         train_size=train_ratio, random_state=random_state)
        x_train = np.array([x.reshape(img_dim,img_dim) for x in x_train])
        x_test = np.array([x.reshape(img_dim,img_dim) for x in x_test])

        if abs(val_ratio) > 0:
            # take fraction of remaining data for validation
            test_ratio = val_ratio / (1-train_ratio)
            x_test, x_val, y_test, y_val = \
            train_test_split(x_test, y_test, test_size=test_ratio, random_state=random_state)
            self.validation_size = len(x_val)
            self.train_size = len(x_train)
            self.test_size = len(x_test)
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)
        
        self.train_size = len(x_train)
        self.test_size = len(x_test)

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

    def data_generator(self, input_img_paths, target_img_paths, cache=False, batch_size=None):
        """Returns a TF Dataset.
        Cache data to memory depending on available GPU memory.
        """
        if batch_size is None:
            batch_size = self.batch_size
        dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
        dataset = dataset.map(self.load_img_masks, num_parallel_calls=tf_data.AUTOTUNE) \
                 .batch(batch_size)
        return dataset.cache() if cache else dataset
    
    def split_data(self, train_ratio = 0.1, val_ratio = 0.1, seed=None, cache=False, weighted_loss=False):
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

        self.train_size = len(self.train_input_paths)
        self.test_size = len(self.test_input_paths)
        self.validation_size = len(self.val_input_paths)

        # Instantiate dataset for each split
        if weighted_loss:

            train_dataset = self.data_generator(self.train_input_paths, self.train_target_paths, batch_size=1)
            valid_dataset = self.data_generator(self.val_input_paths, self.val_target_paths, batch_size=1)
            test_dataset = self.data_generator(self.test_input_paths, self.test_target_paths, batch_size=1)
            train_dataset = self.add_weights(train_dataset, cache)
            valid_dataset = self.add_weights(valid_dataset, cache)
            test_dataset = self.add_weights(test_dataset, cache)

        else:

            train_dataset = self.data_generator(self.train_input_paths, self.train_target_paths, cache)
            valid_dataset = self.data_generator(self.val_input_paths, self.val_target_paths, cache)
            test_dataset = self.data_generator(self.test_input_paths, self.test_target_paths, cache)

        print(f'Data prep. duration: ___{(time.time()-start)/60:.2f}___ minutes.')

        return train_dataset, valid_dataset, test_dataset
    
    
    def add_weights(self, dataset, cache):
        """ Creates class weights for given tf dataset """
        y = np.array([a[-1] for a in dataset.as_numpy_iterator()], dtype='uint8')
        num_pixels = np.prod(y.shape[1:])
        num_class0 = np.count_nonzero(y==0, axis=(1,len(y.shape)-1), keepdims=True)
        num_class1 = np.count_nonzero(y==1, axis=(1,len(y.shape)-1), keepdims=True)
        num_class2 = np.count_nonzero(y==2, axis=(1,len(y.shape)-1), keepdims=True)
        w_class0 = np.divide(num_class0, num_pixels)
        w_class1 = np.divide(num_class1, num_pixels)
        w_class2 = np.divide(num_class2, num_pixels)
        sample_weights = np.where(y==0, w_class0, y)
        sample_weights = np.where((sample_weights==1), w_class1, sample_weights)
        sample_weights = np.where((sample_weights==2), w_class2, sample_weights).astype('float32')

        X = np.array([a[0] for a in dataset.as_numpy_iterator()])
        dset = tf_data.Dataset.from_tensor_slices( (X[0], y[0], sample_weights[0]) ).batch(self.batch_size)

        return dset.cache() if cache else dset
    
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


## ========================================================
# split 3D volume to 2D patches and restore the image from the patches
# reference: https://github.com/anyuzoey/CNNforFaultInterpretation

def split_Image(bigImage,isMask,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number):
#     print(bigImage.shape)
    if isMask==True:
        arr = np.pad(bigImage,((top_pad,bottom_pad),(left_pad,right_pad)),"reflect")
        splits = view_as_windows(arr, (splitsize,splitsize),step=stepsize)
        splits = splits.reshape((vertical_splits_number*horizontal_splits_number,splitsize,splitsize))
    else: 
        arr = np.pad(bigImage,((top_pad,bottom_pad),(left_pad,right_pad),(0,0)),"reflect")
        splits = view_as_windows(arr, (splitsize,splitsize,3),step=stepsize)
        splits = splits.reshape((vertical_splits_number*horizontal_splits_number,splitsize,splitsize,3))
    return splits # return list of arrays.

## restore patches to full image
def recover_Image(patches: np.ndarray, imsize: Tuple[int, int, int], left_pad,right_pad,top_pad,bottom_pad, overlapsize):
#     patches = np.squeeze(patches)
    assert len(patches.shape) == 5

    i_h, i_w, i_chan = imsize
    image = np.zeros((i_h+top_pad+bottom_pad, i_w+left_pad+right_pad, i_chan), dtype=patches.dtype)
    divisor = np.zeros((i_h+top_pad+bottom_pad, i_w+left_pad+right_pad, i_chan), dtype=patches.dtype)
#     print("i_h, i_w, i_chan",i_h, i_w, i_chan)
    n_h, n_w, p_h, p_w,_= patches.shape
    
    o_w = overlapsize
    o_h = overlapsize

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    recover = image / divisor
    return recover[top_pad:top_pad+i_h, left_pad:left_pad+i_w]

## ==================================================
## plot random patch images 

def plot_random_images(images, labels, num_images=4):
    # Select random indices
    num_total_images = len(images)
    random_indices = np.random.choice(num_total_images, size=num_images, replace=False)
    
    fig, axes = plt.subplots(3, num_images, figsize=(18, 10))
    
    # Plot overlapped images
    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(images[idx], cmap='seismic', aspect='auto')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"seismic id: {idx}")
        axes[0, i].imshow(labels[idx], cmap='gray', aspect='auto',alpha=0.3)
        
    # Plot labels
    for i, idx in enumerate(random_indices):
        axes[1, i].imshow(labels[idx],cmap='gray', aspect='auto')
        axes[1, i].set_title(f"label id: {idx}")
        axes[1, i].axis('off')
    
    # Plot original images
    for i, idx in enumerate(random_indices):
        axes[2, i].imshow(images[idx], cmap='seismic', aspect='auto')
        axes[2, i].axis('off')
        axes[2, i].set_title(f"original seismic: {idx}")
    
    plt.tight_layout()
    plt.show()

#====================================================
## Seismic segy header and data loading

# Reference segyio, https://github.com/equinor/segyio
# Reference Segyio-notebook, https://github.com/equinor/segyio-notebooks
# trace header and text header functions

def parse_trace_headers(segyfile, n_traces):
    """
    Parse the segy file trace headers into a pandas dataframe.
    Column names are defined from segyio internal tracefield
    One row per trace
    """
    # Get all header keys
    headers = segyio.tracefield.keys
    # Initialize dataframe with trace id as index and headers as columns
    df = pd.DataFrame(index=range(1, n_traces + 1),columns=headers.keys())
    # Fill dataframe with all header values
    for k, v in headers.items():
        df[k] = segyfile.attributes(v)[:]
        return df

def parse_text_header(segyfile):
    """
    Format segy text header into a readable, clean dict
    """
    raw_header = segyio.tools.wrap(segyfile.text[0])
    # Cut on C*int pattern
    cut_header = re.split(r'C ', raw_header)[1::]
    # Remove end of line return
    text_header = [x.replace('\n', ' ') for x in cut_header]
    text_header[-1] = text_header[-1][:-2]
    # Format in dict
    clean_header = {}
    i = 1
    for item in text_header:
        key = "C" + str(i).rjust(2, '0')
        i += 1
        clean_header[key] = item
    return clean_header

def plot_segy(file):
    """
    Load data and get basic info of no.of samples,traces etc
    possible to check text header and trace header if it is needed
    """
    with segyio.open(file, ignore_geometry=True) as f:
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        data = f.trace.raw[:]
        #Load headers - binary header, text header and trace header
        bin_headers = f.bin
        text_headers = parse_text_header(f)
        trace_headers = parse_trace_headers(f, n_traces)
        print(f'N Traces: {n_traces}, N Samples: {n_samples}, Sample rate: {sample_rate}ms, Trace length: {max(twt)}')
        print(f'2D segy shape: {data.shape}')
        #print(f'Binary header: {bin_headers}')
        #print(f'Text header: {text_headers}')
        #print(f'Trace header: {trace_headers}')
        extent = [1, n_traces, twt[-1], twt[0]]
    return data, extent


## ==================================================

class Thebe(Preprocess):
    """ Handles loading of Thebe seismic data and labels into tf.datasets:
        training, validation and test datasets.
        Data stored as numpy array in npy format in some url (path).
    """
    def __init__(self, 
        seismic_url, 
        annotations_url, 
        threshold=0.5, 
        pos_label=1, 
        img_size=(96,96),
        ratio = None,
        image_cmap='seismic',
        label_cmap='seismic'
    ):
        
        super().__init__(threshold=threshold, pos_label=pos_label)
        
        self.seismic_url = seismic_url
        self.annotations_url = annotations_url
        self.img_size = img_size
        self.ratio = ratio
        self.image_cmap = image_cmap
        self.label_cmap = label_cmap

    def select_path(self, sub_group='train'):
        """ Picks paths containing image patches and 
        annotations for given data subgroup (train, val, test) """

        if sub_group.lower() in ['train','val','test']:
            sub_group = sub_group.lower()
            imgs_path = join(self.seismic_url, sub_group)
            imgs_path = join(imgs_path, f'seis_{sub_group}.npy')
            labels_path = join(self.annotations_url, sub_group)
            labels_path = join(labels_path, f'fault_{sub_group}.npy')
        else: 
            raise Exception(f'Non-valid subgroup: {sub_group}!')

        return imgs_path, labels_path

    def load_img_masks(self, imgs_path, labels_path):
        
        imgs = np.load(imgs_path) #.astype('float32')
        
        labels = np.load(labels_path) #.astype('uint8')
        
        return np.expand_dims(imgs,-1), np.expand_dims(labels,-1)

    
    def data_generator(
        self, 
        sub_group='train', 
        batch_size=1, 
        as_numpy=False, 
        cache=False,
        weighted_loss=False
    ):
        """ 
        Extracts and splits data into training, validation and test sets 
        Cache data to memory depending on available GPU memory.
        """
        imgs_path, labels_path = self.select_path(sub_group)

        X, y = self.load_img_masks(imgs_path, labels_path)

        if isinstance(self.ratio, type(0.0)):
            assert (self.ratio > 0 and self.ratio < 1)
            L = int(len(y) * self.ratio)
            X, y  = X[:L], y[:L]

        if sub_group.__contains__('train'):
            self.train_size = len(y)
        elif sub_group.__contains__('val'):
            self.validation_size = len(y)
        elif sub_group.__contains__('test'):
            self.test_size = len(y)

        if as_numpy:
            return (X, y, self.get_weights(y)) if (weighted_loss) else (X, y)

        dataset = tf_data.Dataset.from_tensor_slices(
            (X, y, self.get_weights(y)) if (weighted_loss) else (X, y)
        )
        dataset = dataset.batch(batch_size)
        
        return dataset.cache() if cache else dataset
    

    
    def get_weights(self, y):
        """ Creates class weights for given labels y """
        num_pixels = np.prod(y.shape[1:])
        num_negatives = np.count_nonzero(abs(y)<1e-6, axis=(1,len(y.shape)-1), keepdims=True)
        num_positives = np.subtract(num_pixels, num_negatives)
        w_positives = np.divide(num_positives, num_pixels)
        w_negatives = np.divide(num_negatives, num_pixels)
        sample_weights = np.where(abs(y)<1e-6, w_negatives, w_positives)

        return sample_weights.astype('float32')
    
    
    def plot_multiple(self, data, idx, label=True):
        num_images = len(idx)
        for col in range(num_images):
            ax = self.axs[self.row][col]
            ax.axis('off')
            if label:
                img = data[col]
                ax.imshow(img, cmap=self.label_cmap)
                ax.set_title(f'label {idx[col]}')
            else:
                img = data[col]
                ax.imshow(img, cmap=self.image_cmap)
                ax.set_title(f'image {idx[col]}')
        self.row = self.row + 1 
                
    def display_sample_images(self, imgs, labels, y_pred=None, num_images=8, save_path=None):
        """Display images, labels and predictions for a selected number of images."""
        idx = sorted(
            np.random.choice(np.arange(len(labels)), num_images, replace=False)
        )
        
        if isinstance(save_path, str):
            with open(join(save_path,'images.npy'), 'wb') as fp:
                np.save(fp, imgs[idx])
            with open(join(save_path,'labels.npy'), 'wb') as fp:
                np.save(fp, labels[idx])
            with open(join(save_path,'index.npy'), 'wb') as fp:
                np.save(fp, idx)

        self.num_rows = 2 if (y_pred is None) else 3
        fig, axs = plt.subplots(ncols=num_images, nrows=self.num_rows)
        fig.set_figwidth(15)
        self.axs = axs
        self.row = 0
        
        # Display input image
        self.plot_multiple(imgs[idx], idx, label=False)
        
        # Display ground-truth target mask
        self.plot_multiple(labels[idx], idx)

        # Display mask predicted by our model
        if (y_pred is not None):
            self.plot_multiple(y_pred[idx], idx)

        fig.tight_layout(pad=0.0)
        plt.subplots_adjust(hspace=0.5)
        plt.show()
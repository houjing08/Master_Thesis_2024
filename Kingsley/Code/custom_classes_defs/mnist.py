import struct
from array import array
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#
# MNIST Data Loader Class: From Keras-io
#
class MNIST_digits(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
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
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (np.array(x_train), np.array(y_train)),(np.array(x_test), np.array(y_test))        

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

class MNIST_letters(object):
    """ Handles loading, scaling and preprocessing of MNIST handwritten letters """
    def __init__(self, url, edge_letters=True):

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
        self.scaler = MinMaxScaler().fit([[min_val],[max_val]])

    def split_data(self, train_ratio = 0.1, img_dim=28, random_state=1):
        """ Split data into train-test proportions, given train-ratio, and re-arrange"""
        x_train, x_test, y_train, y_test = \
        train_test_split(self.df.values, self.df.index.values, train_size=0.1, random_state=random_state)
        x_train = np.array([x.reshape(img_dim,img_dim) for x in x_train])
        x_test = np.array([x.reshape(img_dim,img_dim) for x in x_test])

        return (x_train, y_train), (x_test, y_test)

    def transform(self, X):
        return self.scaler.transform(X.reshape(-1,1)).reshape(X.shape)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X.reshape(-1,1)).reshape(X.shape)

    def normalize(self, data):
        data = np.expand_dims(data, -1).astype("float32")
        return self.transform(data)

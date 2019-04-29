import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class MNIST:
    '''
    Load and configure the MNIST dataset for training.
    '''
    def __init__(self, batch_size, normalize_data,
                 one_hot_encoding, flatten_images,
                 shuffle_per_epoch):
        self.train_x, self.train_y_cls, self.test_x, self.test_y_cls = self.__load_data()
        self.train_y_cls = self.train_y_cls.astype(np.int64)
        self.test_y_cls = self.test_y_cls.astype(np.int64)
        self._batch_size = batch_size
        self._shuffle_per_epoch = shuffle_per_epoch
        self.n_train_batches = self.train_x.shape[0]//self._batch_size
        self.n_test_batches = self.test_x.shape[0]//self._batch_size
        self.original_image_shape = self.return_original_image_shape()

        self._normalize_data = normalize_data
        if self._normalize_data:
            self.__normalize_images()

        self._flatten_images = flatten_images
        if self._flatten_images:
            self.__flatten_sets()

        self._one_hot_encoding = one_hot_encoding
        if self._one_hot_encoding:
            self.train_y, self.test_y = self.__one_hot_encoding()

    def __iter__(self):
        self.__index = 0
        if self._shuffle_per_epoch:
            self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
        self.chucked_train_x = np.array_split(self.train_x, self.n_train_batches)
        self.chucked_test_x = np.array_split(self.train_y, self.n_train_batches)
        return self

    def __next__(self):
        if self.__index == len(self):
            raise StopIteration()

        batch_x, batch_y = self.chucked_train_x[self.__index], self.chucked_test_x[self.__index]
        self.__index += 1

        return (batch_x, batch_y)

    def __len__(self):
        return self.n_train_batches

    def return_num_classes(self):
        '''
        Return the number of classes present in the dataset.
        '''
        unique_labels = np.unique(self.test_y_cls)
        num_classes = unique_labels.shape[0]

        return num_classes

    def return_original_image_shape(self):
        '''
        Return the original shape of the MNIST images.
        '''
        random_sample = self.test_x[0]

        return (random_sample.shape[0], random_sample.shape[1])

    def return_input_shape(self):
        '''
        Return the shape of the dataset dimensions.
        '''
        random_sample = self.test_x[0]
        input_dimensions_shape = random_sample.shape[0]

        return input_dimensions_shape

    def __load_data(self):
        '''
        Load the MNIST dataset.
        '''
        train_set, test_set = tf.keras.datasets.mnist.load_data()
        train_x, train_y = train_set
        test_x, test_y = test_set

        return train_x, train_y, test_x, test_y

    def __normalize_images(self):
        '''
        Normalize the image pixels to facilitate training.
        '''
        self.train_x = self.train_x/255
        self.test_x = self.test_x/255

    def __one_hot_encoding(self):
        '''
        Retrieve the one_hot encodings from int labels.
        '''
        train_y = tf.keras.utils.to_categorical(self.train_y_cls)
        test_y = tf.keras.utils.to_categorical(self.test_y_cls)

        return train_y, test_y

    def __flatten_sets(self):
        '''
        Flatten the images from [None, 28, 28] to [None, 784]
        '''
        self.train_x = np.reshape(self.train_x,
                                  [self.train_x.shape[0],
                                   self.train_x.shape[1]*self.train_x.shape[2]])
        self.test_x = np.reshape(self.test_x,
                                 [self.test_x.shape[0],
                                  self.test_x.shape[1]*self.test_x.shape[2]])

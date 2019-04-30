import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

class CIFAR10:
    '''
    Load and configure the CIFAR dataset for training.
    '''
    def __init__(self, batch_size, normalize_data,
                 one_hot_encoding, flatten_images,
                 shuffle_per_epoch, crop_size,
                 image_preprocessing):
        self.train_x, self.train_y_int, self.test_x, self.test_y_int = self.__load_data()
        self.train_y_int = self.train_y_int.astype(np.int64)
        self.test_y_int = self.test_y_int.astype(np.int64)
        self._label_encoder = self.__init_label_encoder()
        self.train_y_cls = self.encode_classes(self.train_y_int)
        self.test_y_cls = self.encode_classes(self.test_y_int)
        self._batch_size = batch_size
        self._shuffle_per_epoch = shuffle_per_epoch
        self.n_train_batches = self.train_x.shape[0]//self._batch_size
        self.n_test_batches = self.test_x.shape[0]//self._batch_size
        self.original_image_shape = self.return_original_image_shape()
        self._img_size_cropped = crop_size
        self._num_channels = self.return_num_channels()

        if normalize_data:
            self.__normalize_images()

        if image_preprocessing:
            self.train_x, self.test_x = self.__apply_pre_processing()

        if flatten_images:
            self.__flatten_sets()

        if one_hot_encoding:
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
        unique_labels = np.unique(self.test_y_int)
        num_classes = unique_labels.shape[0]

        return num_classes

    def return_original_image_shape(self):
        '''
        Return the original shape of the MNIST images.
        '''
        random_sample = self.test_x[0]

        return (random_sample.shape[0], random_sample.shape[1])

    def return_num_channels(self):
        '''
        Return the number of channels (colours) of the
        dataset.
        '''
        random_sample = self.test_x[0]

        return random_sample.shape[-1]

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
        train_set, test_set = tf.keras.datasets.cifar10.load_data()
        train_x, train_y = train_set
        test_x, test_y = test_set

        return train_x, train_y.flatten(), test_x, test_y.flatten()

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
        train_y = tf.keras.utils.to_categorical(self.train_y_int)
        test_y = tf.keras.utils.to_categorical(self.test_y_int)

        return train_y, test_y

    def __init_label_encoder(self):
        '''
        Initialise the label encoder and fit to classes.
        '''
        # Existing classes
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        # Fit classes to encoder.
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)

        return label_encoder

    def encode_classes(self, data):
        '''
        Encode integers to class strings.
        '''

        encoded_data = self._label_encoder.inverse_transform(data)

        return encoded_data

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

    def __pre_process_image(self, image, training):
        '''
        Perform a random pre-processing on a single
        image in order to make the system more robust
        to intricacies.
        '''
        # Randomly crop the image
        if training:
            image = tf.random_crop(image, size=[self._img_size_cropped,
                                                self._img_size_cropped,
                                                self._num_channels])
            # Randomly flip the image
            image = tf.image.random_flip_left_right(image)
            # Randomly adjust aspects of the image
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
            # Limit the pixel values
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 1.0)

        else:
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=self._img_size_cropped,
                                                           target_width=self._img_size_cropped)

        return image

    def __apply_pre_processing(self):
        '''
        Apply image preprocessing to train set.
        '''
        train_x = tf.map_fn(lambda image: self.__pre_process_image(image, True),
                            self.train_x)
        test_x = tf.map_fn(lambda image: self.__pre_process_image(image, False),
                           self.test_x)

        return train_x, test_x

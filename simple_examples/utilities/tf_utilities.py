'''
Helper classes and functions for tensorflow operations.
'''
import tensorflow as tf
import numpy as np

def corrupt_data_with_noise(input_data_shape, corruption_level):
    '''
    Corrupt the input data to be used in an autoencoder
    model.
    '''
    noise = np.random.binomial(1, 1 - corruption_level, input_data_shape)

    return noise

def get_layer_output(session, graph_input, input_data, layer):
    '''
    Retrieve a layer output for one input instance.
    '''
    feed_dict = {graph_input: input_data}
    output = session.run(layer, feed_dict=feed_dict)

    return output

def transform_tf_variable(session, variable):
    '''
    Transform any tensorflow variable to a standard numpy array.
    '''
    transformed_variable = session.run(variable)

    return transformed_variable

def new_weights(shape):
    '''
    Construct weights for layer.
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    '''
    Construct bias for layer.
    '''
    return tf.Variable(tf.constant(0.05, shape=[length]))

class Conv2DLayer():
    '''
    Simple class for defining convolutional layers.
    '''
    def __init__(self, input_data, num_input_channels,
                 filter_size, num_filters, strides=[1, 1, 1, 1],
                 padding='SAME'):
        # Define class variables
        self._input = input_data
        self._shape = [filter_size, filter_size, num_input_channels, num_filters]
        self._strides = strides
        self._padding = padding
        # Get weights, biases and build conv layer
        self.weights = new_weights(shape=self._shape)
        self._biases = new_biases(length=num_filters)
        self._layer = self.__build_convolutional_layer()
        self.output_layer = self.__get_layer_output()

    def __build_convolutional_layer(self):
        '''
        Call tensorflow's convolution function and get
        the layer values.
        '''
        layer = tf.nn.conv2d(input=self._input,
                             filter=self.weights,
                             strides=self._strides,
                             padding=self._padding)
        return layer

    def __get_layer_output(self):
        '''
        Get the output of the convolution layer by adding
        the biases.
        '''
        layer_output = self._layer + self._biases

        return layer_output

class MaxPoolLayer():
    '''
    Simple class for defining max pooling layers.
    '''
    def __init__(self, input_layer, ksize=[1, 2, 2, 1],
                 strides=[1, 2, 2, 1], padding='SAME',
                 activation=tf.nn.relu):
        self._input = input_layer
        self._strides = strides
        self._ksize = ksize
        self._padding = padding
        self._activation = activation

        # Build and activate the layer
        self._layer = self.__build_max_pool_layer()
        self.output_layer = self.__activate_layer()

    def __build_max_pool_layer(self):
        '''
        Build and return the layer.
        '''
        layer = tf.nn.max_pool(value=self._input,
                               ksize=self._ksize,
                               strides=self._strides,
                               padding=self._padding)

        return layer

    def __activate_layer(self):
        '''
        Activate the output of the max pool layer.
        '''
        output_layer = self._activation(self._layer)

        return output_layer

class DenseLayer():
    '''
    Simple dense layer class.
    '''
    def __init__(self, input_data,
                 num_inputs, num_outputs,
                 activation=tf.nn.relu):
        self._input = input_data
        self._activation = activation
        self.weights = new_weights(shape=[num_inputs, num_outputs])
        self._biases = new_biases(length=num_outputs)
        self.pre_activation_layer = self.__build_dense_layer()
        self.output_layer = self.__activate_layer()

    def __build_dense_layer(self):
        '''
        Construct the dense layer.
        '''
        layer = tf.matmul(self._input, self.weights) + self._biases

        return layer

    def __activate_layer(self):
        '''
        Activate the layer.
        '''
        output_layer = self._activation(self.pre_activation_layer)

        return output_layer

class AutoencoderLayer():
    '''
    Simple autoencoder layer class.
    '''
    def __init__(self, input_data, num_inputs, num_outputs,
                 encoder_activation=tf.nn.relu,
                 decoder_activation=tf.nn.sigmoid,
                 activate=True):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._input = input_data
        self.mask = self.__noise_mask(num_inputs)
        self.corrupted_input = self.__corrupt_input()
        self.weights, self._biases, \
        self.weights_prime, self._biases_prime = self.__init_weights(num_inputs,
                                                                     num_outputs)
        self._activate = activate
        self._encoder_activation = encoder_activation
        self._decoder_activation = decoder_activation
        self.encoded_input = self.encoder(self.corrupted_input)
        self.decoded_input = self.decoder(self.encoded_input)
        self.cost = self.__calculate_cost()

    def __noise_mask(self, num_inputs):
        '''
        Define the noise mask placeholder for the layer.
        '''
        mask = tf.placeholder(tf.float32, [None, num_inputs])

        return mask

    def __corrupt_input(self):
        '''
        Add noise to the input of the autoencoder.
        '''
        corrupted_input = self.mask * self._input

        return corrupted_input

    def __init_weights(self, num_inputs, num_outputs):
        '''
        Construct the weights for the autoencoder layer,
        both encoding and decoding.
        '''
        weights = new_weights(shape=[num_inputs, num_outputs])
        biases = new_biases(length=num_outputs)

        weights_prime = tf.transpose(weights)
        biases_prime = new_biases(length=num_inputs)

        return weights, biases, weights_prime, biases_prime

    def encoder(self, input_x):
        '''
        Encoder part of the system to reduce the dimensionality
        of the input data.
        '''
        linear_activation = tf.matmul(input_x, self.weights) + self._biases
        if self._activate:
            encoded_input = self._encoder_activation(linear_activation)
        else:
            encoded_input = linear_activation

        return encoded_input

    def decoder(self, input_x):
        '''
        Decoder part of the system to reconstruct the output
        of the encoder.
        '''
        linear_activation = tf.matmul(input_x, self.weights_prime) + self._biases_prime
        if self._activate:
            decoded_input = self._decoder_activation(linear_activation)
        else:
            decoded_input = linear_activation

        return decoded_input

    def __calculate_cost(self):
        cost = tf.reduce_sum(tf.pow(self._input - self.decoded_input, 2))

        return cost

class FlattenLayer():
    '''
    Simple class to flatten the conv layer.
    '''
    def __init__(self, input_layer):
        self._input = input_layer
        self._input_shape = self._input.get_shape()
        self.num_features = self._input_shape[1:].num_elements()

        self.output_layer = self.__flatten_layer()

    def __flatten_layer(self):
        '''
        Flatten and return the input layer.
        '''
        layer_flat = tf.reshape(self._input, [-1, self.num_features])

        return layer_flat

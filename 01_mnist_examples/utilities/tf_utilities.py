import tensorflow as tf

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
                 padding='SAME', use_pooling=True):
        # Define class variables
        self._input = input_data
        self._shape = [filter_size, filter_size, num_input_channels, num_filters]
        self._strides = strides
        self._padding = padding
        # Get weights, biases and build conv layer
        self._weights = new_weights(shape=self._shape)
        self._biases = new_biases(length=num_filters)
        self._layer = self.__build_convolutional_layer()
        self.output_layer = self.__get_layer_output()

    def __build_convolutional_layer(self):
        '''
        Call tensorflow's convolution function and get
        the layer values.
        '''
        layer = tf.nn.conv2d(input=self._input,
                             filter=self._weights,
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
        self._weights = new_weights(shape=[num_inputs, num_outputs])
        self._biases = new_biases(length=num_outputs)
        self.pre_activation_layer = self.__build_dense_layer()
        self.output_layer = self.__activate_layer()

    def __build_dense_layer(self):
        '''
        Construct the dense layer.
        '''
        layer = tf.matmul(self._input, self._weights) + self._biases

        return layer

    def __activate_layer(self):
        '''
        Activate the layer.
        '''
        output_layer = self._activation(self.pre_activation_layer)

        return output_layer

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

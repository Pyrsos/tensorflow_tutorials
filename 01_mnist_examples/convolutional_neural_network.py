import tensorflow as tf
from mnist import MNIST

# Convolutional layer 1
filter_size1 = 5
num_filters1 = 16

# Convolutional layer 2
filter_size2 = 5
num_filters2 = 36

# Fully connected layer.
fc_size = 128

# Import data
mnist = MNIST(batch_size=100, normalize_data=True,
                one_hot_encoding=True, flatten_images=True,
                shuffle_per_epoch=True)

# Data information
img_size = mnist.original_image_shape
img_size_flat = mnist.return_input_shape()
num_classes = mnist.return_num_classes()
num_channels = 1 # Because this is greyscale (need to introduce in the MNIST class)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

class Conv2DLayer():
    '''
    Simple class for defining convolutional layers.
    '''
    def __init__(self, input_data, num_input_channels,
                filter_size, num_filters, strides=[1, 1, 1, 1],
                padding = 'SAME', use_pooling=True):
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
        self._num_features = self._input_shape[1:4].num_elements()

        self.output_layer = self.__flatten_layer()

    def __flatten_layer(self):
        '''
        Flatten and return the input layer.
        '''
        layer_flat = tf.reshape(self._input, [-1, self._num_features])

        return layer_flat

# Placeholders

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size[0], img_size[1], num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# First convolutional layer
conv_1 = Conv2DLayer(input_data=x_image, num_input_channels=num_channels,
                     filter_size=filter_size1, num_filters=num_filters1)
max_pool1 = MaxPoolLayer(input_layer=conv_1.output_layer)
# Second convolutional layer
conv_2 = Conv2DLayer(input_data=max_pool1.output_layer, num_input_channels=num_filters1,
                     filter_size=filter_size2, num_filters=num_filters2)
max_pool2 = MaxPoolLayer(input_layer=conv_2.output_layer)
# Flatten layer and pass to dense layer
layer_flat = FlattenLayer(input_layer=max_pool2.output_layer)
dense_1 = DenseLayer(input_data=layer_flat.output_layer,
                     num_inputs=layer_flat._num_features, 
                     num_outputs=fc_size)
# Softmax layer
dense_2 = DenseLayer(input_data=dense_1.output_layer,
                     num_inputs=fc_size, num_outputs=num_classes,
                     activation=tf.nn.softmax)

# Predictions
y_pred = dense_2.output_layer
y_pred_cls = tf.argmax(y_pred, axis=1)

# Loss
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_true,
                                                logits=dense_2.pre_activation_layer)
cost = tf.reduce_mean(cross_entropy)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Performance measure
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run session
session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(1, 101):
    for j, (batch_x, batch_y) in enumerate(mnist):
        feed_dict_train = {x: batch_x, y_true: batch_y}
        _, loss = session.run([optimizer, cost], feed_dict=feed_dict_train)
        print("Epoch {}/{}, Batch: {}/{} with Loss: {}".format(
            i, 100, j, len(mnist), loss))

    feed_dict_test = {x: mnist.test_x, y_true: mnist.test_y, y_true_cls: mnist.test_y_cls}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Epoch {}/{}, Accuracy: {}".format(i, 100, acc))




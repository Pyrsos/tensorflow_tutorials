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

def new_conv_layer(input, num_input_channels,
                   filter_size, num_filters,
                   use_pooling=True):
    # Shape of filter weights for convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    # Based on the shape create weights and biases
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    # Create the conv layer (as it is an image this will be 2d)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    # Add the biases to the layer
    layer += biases

    # Use maxpooling to down-sample image resolution
    if use_pooling:
        # This is a 2x2 max pooling. This means that we consider
        # 2x2 windows and select the largest value in each window.
        # Then we move 2 pixels to the next window. 
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Activate the layer using relu. Relu basically calculates
    # max(x, 0) for each pixel x.
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    # Get the shape of the layer. This is assumed to be
    # [batch_size, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()
    # Therefore the number of features will be img_height*img_width*num_channels
    num_features = layer_shape[1:4].num_elements()
    # Then reshape (flatten) the layer
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input, num_inputs,
                 num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# Placeholders

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size[0], img_size[1], num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Build model layers and connect

layer_conv1, weights_conv1 = new_conv_layer(
    input=x_image, num_input_channels=num_channels,
    filter_size=filter_size1, num_filters=num_filters1,
    use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(
    input=layer_conv1, num_input_channels=num_filters1,
    filter_size=filter_size2, num_filters=num_filters2,
    use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(
    input=layer_flat, num_inputs=num_features,
    num_outputs=fc_size, use_relu=True)

layer_fc2 = new_fc_layer(
    input=layer_fc1, num_inputs=fc_size,
    num_outputs=num_classes, use_relu=False)

# Predictions

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Loss
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_true,
                                                logits=layer_fc2)
cost = tf.reduce_mean(cross_entropy)

import ipdb; ipdb.set_trace()
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
            i, 101, j, len(mnist), loss))

    feed_dict_test = {x: mnist.test_x, y_true: mnist.test_y, y_true_cls: mnist.test_y_cls}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Epoch {}/{}, Accuracy: {}".format(i, 101, acc))




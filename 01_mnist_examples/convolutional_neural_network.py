import tensorflow as tf
from mnist import MNIST
from tf_utilities import new_weights, new_biases, Conv2DLayer, MaxPoolLayer, DenseLayer, FlattenLayer

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
epochs = 1

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(1, epochs+1):
    for j, (batch_x, batch_y) in enumerate(mnist):
        feed_dict_train = {x: batch_x, y_true: batch_y}
        _, loss = session.run([optimizer, cost], feed_dict=feed_dict_train)
        print("Epoch {}/{}, Batch: {}/{} with Loss: {}".format(
            i, epochs, j, len(mnist), loss))

    feed_dict_test = {x: mnist.test_x, y_true: mnist.test_y, y_true_cls: mnist.test_y_cls}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Epoch {}/{}, Accuracy: {}".format(i, epochs, acc))

import ipdb; ipdb.set_trace()


import matplotlib.pyplot as plt
from absl import flags
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from dataset_utilities.mnist import MNIST
from utilities.tf_utilities import Conv2DLayer, MaxPoolLayer, DenseLayer, FlattenLayer, transform_tf_variable, get_layer_output
from utilities.plot_utilities import plot_conv_weights, plot_conv_layer, print_confusion_matrix, find_wrong_predictions, plot_images

flags.DEFINE_integer("filter_size_1", 5, help="Size of the filters for the first conv layer")
flags.DEFINE_integer("num_filters_1", 16, help="Number of filters in first conv layer")
flags.DEFINE_integer("filter_size_2", 5, help="Size of the filters for the second conv layer")
flags.DEFINE_integer("num_filters_2", 36, help="Number of filters in second conv layer")
flags.DEFINE_integer("fc_size", 128, help="Size of fully connected layer")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("epochs", 100, help="Number of epochs")
flags.DEFINE_float("learning_rate", 1e-4, help="Learning rate")

FLAGS = flags.FLAGS

def main(_):
    '''
    Main body of code for creating a two-layer convolutional neural
    network, with max pooling, flatten dense layer and softmax output.
    '''
    # Import data
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                  one_hot_encoding=True, flatten_images=False,
                  shuffle_per_epoch=True)

    # Data information
    img_size = mnist.original_image_shape
    num_classes = mnist.return_num_classes()
    num_channels = 1 # Because this is greyscale (need to introduce in the MNIST class)

    # Placeholders
    x_input = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1]], name='x_input')
    x_image = tf.reshape(x_input, shape=[-1, img_size[0], img_size[1], num_channels])

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    # First convolutional layer
    conv_1 = Conv2DLayer(input_data=x_image, num_input_channels=num_channels,
                         filter_size=FLAGS.filter_size_1, num_filters=FLAGS.num_filters_1)
    max_pool1 = MaxPoolLayer(input_layer=conv_1.output_layer)
    # Second convolutional layer
    conv_2 = Conv2DLayer(input_data=max_pool1.output_layer, num_input_channels=FLAGS.num_filters_1,
                         filter_size=FLAGS.filter_size_2, num_filters=FLAGS.num_filters_2)
    max_pool2 = MaxPoolLayer(input_layer=conv_2.output_layer)
    # Flatten layer and pass to dense layer
    layer_flat = FlattenLayer(input_layer=max_pool2.output_layer)
    dense_1 = DenseLayer(input_data=layer_flat.output_layer,
                         num_inputs=layer_flat.num_features,
                         num_outputs=FLAGS.fc_size)
    # Softmax layer
    softmax_layer = DenseLayer(input_data=dense_1.output_layer,
                               num_inputs=FLAGS.fc_size, num_outputs=num_classes,
                               activation=tf.nn.softmax)

    # Predictions
    y_pred = softmax_layer.output_layer
    y_pred_cls = tf.argmax(y_pred, axis=1)

    # Loss
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_true,
                                                    logits=softmax_layer.pre_activation_layer)
    cost = tf.reduce_mean(cross_entropy)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

    # Performance measure
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    loss = 0
    acc = 0

    with tqdm(total=FLAGS.epochs, postfix='Accuracy = {:.3f}'.format(acc)) as epoch_progress:
        for epoch in range(FLAGS.epochs):
            with tqdm(total=len(mnist), postfix='Loss: {:.3f}'.format(loss),
                      mininterval=1e-4, leave=True) as batch_progress:
                # Train model per batch
                for batch_x, batch_y in mnist:
                    feed_dict_train = {x_input: batch_x, y_true: batch_y}
                    _, loss = session.run([optimizer, cost], feed_dict=feed_dict_train)
                    batch_progress.set_postfix(Loss=loss)
                    batch_progress.update()

                # Test model for entire test set
                feed_dict_test = {x_input: mnist.test_x,
                                  y_true: mnist.test_y,
                                  y_true_cls: mnist.test_y_cls}

                acc, predictions = session.run([accuracy, y_pred_cls], feed_dict=feed_dict_test)
                epoch_progress.set_postfix(Accuracy=acc)
                epoch_progress.update()

    # Plot the filter weights for the convolutional layers
    conv_1_weights = transform_tf_variable(session, conv_1._weights)
    conv_2_weights = transform_tf_variable(session, conv_2._weights)
    plot_conv_weights(conv_1_weights)
    plot_conv_weights(conv_2_weights)
    # Plot the outputs of the convolutional layers.
    conv_1_layer_output = get_layer_output(session, x_input, [mnist.test_x[0]], conv_1.output_layer)
    conv_2_layer_output = get_layer_output(session, x_input, [mnist.test_x[0]], conv_2.output_layer)
    plot_conv_layer(conv_1_layer_output)
    plot_conv_layer(conv_2_layer_output)
    # Plot the confusion matrix
    print_confusion_matrix(labels=mnist.test_y_cls,
                           predictions=predictions,
                           num_classes=num_classes)
    # Plot images that are classified incorrectly
    wrong_images, wrong_labels, correct_labels = find_wrong_predictions(labels=mnist.test_y_cls,
                                                                        predictions=predictions,
                                                                        images=mnist.test_x)
    incorrect_logits = get_layer_output(session, x_input, wrong_images, softmax_layer.pre_activation_layer)
    incorrect_predictions = get_layer_output(session, x_input, wrong_images, softmax_layer.output_layer)
    plot_images(images=wrong_images[:5], y_pred=incorrect_predictions[:5], logits=incorrect_logits[:5],
                cls_true=correct_labels[:5], cls_pred=wrong_labels[:5],
                img_shape=img_size)

if __name__ == '__main__':
    tf.app.run()

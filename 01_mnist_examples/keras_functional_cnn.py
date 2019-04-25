'''
Example script for constructing CNN for classifying the MNIST dataset using keras.
'''
from absl import flags
import numpy as np
import tensorflow as tf
from dataset_utilities.mnist import MNIST
from utilities.plot_utilities import print_confusion_matrix, find_wrong_predictions, plot_images

flags.DEFINE_integer("filter_size_1", 5, help="Size of the filters for the first conv layer")
flags.DEFINE_integer("num_filters_1", 16, help="Number of filters in first conv layer")
flags.DEFINE_integer("filter_size_2", 5, help="Size of the filters for the second conv layer")
flags.DEFINE_integer("num_filters_2", 36, help="Number of filters in second conv layer")
flags.DEFINE_integer("fc_size", 128, help="Size of fully connected layer")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("epochs", 100, help="Number of epochs")
flags.DEFINE_float("learning_rate", 1e-4, help="Learning rate")
flags.DEFINE_float("dropout_rate", 0.2, help="Dropout rate")

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

    # Construct the model
    inputs = tf.keras.layers.Input(shape=(img_size[0], img_size[1]))
    net = inputs
    net = tf.keras.layers.Reshape((img_size[0], img_size[1], num_channels))(net)
    net = tf.keras.layers.Conv2D(kernel_size=FLAGS.filter_size_1, strides=1,
                                 filters=FLAGS.num_filters_1, padding='same',
                                 activation='relu', name='conv_layer1')(net)
    net = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(net)
    net = tf.keras.layers.Dropout(FLAGS.dropout_rate)(net)
    net = tf.keras.layers.Conv2D(kernel_size=FLAGS.filter_size_2, strides=1,
                                 filters=FLAGS.num_filters_2, padding='same',
                                 activation='relu', name='conv_layer2')(net)
    net = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(net)
    net = tf.keras.layers.Dropout(FLAGS.dropout_rate)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(FLAGS.fc_size, activation='relu')(net)
    net = tf.keras.layers.Dropout(FLAGS.dropout_rate)(net)
    net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)

    outputs = net
    # Construct keras model by passing inputs and outputs to the graph
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # Optimization and cost
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Training
    model.fit(x=mnist.train_x, y=mnist.train_y,
              epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
    # Evaluation
    model.evaluate(x=mnist.test_x, y=mnist.test_y)
    logits = model.predict(x=mnist.test_x)
    predictions = np.argmax(logits, axis=1)

    # Plot the confusion matrix
    print_confusion_matrix(labels=mnist.test_y_cls,
                           predictions=predictions,
                           num_classes=num_classes)
    # Plot images that are classified incorrectly
    wrong_images, wrong_labels, correct_labels = find_wrong_predictions(labels=mnist.test_y_cls,
                                                                        predictions=predictions,
                                                                        images=mnist.test_x)
    incorrect_logits = model.predict(wrong_images)
    # Need to find a way to get the output on the different layers of the
    # model. Here, the pre-activation dense layer values are necessary for
    # plotting.
    plot_images(images=wrong_images[:5], y_pred=incorrect_logits[:5],
                logits=incorrect_logits[:5], cls_true=correct_labels[:5],
                cls_pred=wrong_labels[:5], img_shape=img_size)

    # Save model
    path_model = 'model.keras'
    model.save(path_model)

if __name__ == '__main__':
    tf.app.run()

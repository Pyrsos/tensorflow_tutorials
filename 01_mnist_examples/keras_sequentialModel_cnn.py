'''
Example script for constructing CNN for classifying the MNIST dataset using keras.
'''
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer, Input, Reshape, MaxPooling2D,
                                     Conv2D, Dense, Flatten, Dropout)
from tensorflow.python.keras.optimizers import Adam
from dataset_utilities.mnist import MNIST
from utilities.plot_utilities import (plot_conv_weights, plot_conv_layer, print_confusion_matrix,
                                      find_wrong_predictions, plot_images)

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
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size[0], img_size[1])))
    model.add(Reshape((img_size[0], img_size[1], num_channels)))
    model.add(Conv2D(kernel_size=FLAGS.filter_size_1, strides=1,
                     filters=FLAGS.num_filters_1, padding='same',
                     activation='relu', name='conv_layer1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(FLAGS.dropout_rate))
    model.add(Conv2D(kernel_size=FLAGS.filter_size_1, strides=1,
                     filters=FLAGS.num_filters_1, padding='same',
                     activation='relu', name='conv_layer2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(FLAGS.dropout_rate))
    model.add(Flatten())
    model.add(Dense(FLAGS.fc_size, activation='relu'))
    model.add(Dropout(FLAGS.dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Optimization and cost for compiling the model
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
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

if __name__=='__main__':
    tf.app.run()

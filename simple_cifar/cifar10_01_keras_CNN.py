'''
Example script for constructing CNN for classifying the CIFAR10 dataset using keras.
'''
from absl import flags
import numpy as np
import tensorflow as tf
from pyrsostf.data import CIFAR10
from pyrsostf.plots import print_confusion_matrix, find_wrong_predictions_cifar, plot_images

flags.DEFINE_integer("num_filters_1", 32, help="Number of filters in first conv layer")
flags.DEFINE_integer("num_filters_2", 32, help="Number of filters in second conv layer")
flags.DEFINE_integer("num_filters_3", 64, help="Number of filters in first conv layer")
flags.DEFINE_integer("num_filters_4", 64, help="Number of filters in second conv layer")
flags.DEFINE_integer("fc_size_1", 512, help="Size of first fully connected layer")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("epochs", 100, help="Number of epochs")
flags.DEFINE_float("learning_rate", 1e-4, help="Learning rate")
flags.DEFINE_float("conv_dropout_rate", 0.25, help="Dropout rate")
flags.DEFINE_float("dense_dropout_rate", 0.5, help="Dropout rate")

FLAGS = flags.FLAGS

def main(_):
    '''
    Main body of code for creating a four-layer convolutional neural
    network, with max pooling, flatten dense layer and softmax output.
    '''
    # Import data
    cifar10 = CIFAR10(batch_size=FLAGS.batch_size, normalize_data=True,
                    one_hot_encoding=True, flatten_images=False,
                    shuffle_per_epoch=True, crop_size=24, image_preprocessing=False)

    # Data information
    img_size = cifar10.original_image_shape
    num_classes = cifar10.return_num_classes()
    num_channels = 3

    # Construct the model
    inputs = tf.keras.layers.Input(shape=(img_size[0], img_size[1], num_channels))
    net = inputs
    net = tf.keras.layers.Conv2D(kernel_size=(3, 3),
                                 filters=FLAGS.num_filters_1, padding='same',
                                 activation='relu', name='conv_layer1')(net)
    net = tf.keras.layers.Conv2D(kernel_size=(3, 3),
                                 filters=FLAGS.num_filters_2, padding='same',
                                 activation='relu', name='conv_layer2')(net)
    net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = tf.keras.layers.Dropout(FLAGS.conv_dropout_rate)(net)

    net = tf.keras.layers.Conv2D(kernel_size=(3, 3),
                                 filters=FLAGS.num_filters_3, padding='same',
                                 activation='relu', name='conv_layer3')(net)
    net = tf.keras.layers.Conv2D(kernel_size=(3, 3),
                                 filters=FLAGS.num_filters_4, padding='same',
                                 activation='relu', name='conv_layer4')(net)
    net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = tf.keras.layers.Dropout(FLAGS.conv_dropout_rate)(net)

    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(FLAGS.fc_size_1, activation='relu')(net)
    net = tf.keras.layers.Dropout(FLAGS.dense_dropout_rate)(net)
    net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)

    outputs = net
    # Construct keras model by passing inputs and outputs to the graph
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # Optimization and cost
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Callbacks for saving and early stopping
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                 tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                    monitor='val_loss',
                                                    save_best_only=True)]
    # Training
    model.fit(x=cifar10.train_x, y=cifar10.train_y,
              epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
              callbacks=callbacks,
              validation_data=(cifar10.test_x, cifar10.test_y))
    # Evaluation
    model.evaluate(x=cifar10.test_x, y=cifar10.test_y)
    logits = model.predict(x=cifar10.test_x)
    predictions = np.argmax(logits, axis=1)
    predictions = cifar10.encode_classes(predictions)

    # Plot the confusion matrix
    print_confusion_matrix(labels=cifar10.test_y_cls,
                           predictions=predictions,
                           num_classes=num_classes)
    # Plot images that are classified incorrectly
    wrong_indeces, wrong_labels, correct_labels = find_wrong_predictions_cifar(labels=cifar10.test_y_cls,
                                                                               predictions=predictions)

    wrong_images = cifar10.test_x[wrong_indeces[:10]]
    incorrect_logits = model.predict(wrong_images, batch_size=FLAGS.batch_size)
    # Need to find a way to get the output on the different layers of the
    # model. Here, the pre-activation dense layer values are necessary for
    # plotting.
    plot_images(images=wrong_images[:5], y_pred=incorrect_logits[:5],
                logits=incorrect_logits[:5], cls_true=correct_labels[:5],
                cls_pred=wrong_labels[:5], img_shape=img_size,
                reshape=False)

if __name__ == '__main__':
    tf.app.run()

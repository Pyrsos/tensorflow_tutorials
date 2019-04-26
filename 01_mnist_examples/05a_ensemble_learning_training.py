'''
Example script for constructing an ensemble of CNNs for classifying the MNIST dataset using keras.
'''
from absl import flags
from sklearn.model_selection import train_test_split
import tensorflow as tf
from dataset_utilities.mnist import MNIST

flags.DEFINE_integer("filter_size_1", 5, help="Size of the filters for the first conv layer")
flags.DEFINE_integer("num_filters_1", 16, help="Number of filters in first conv layer")
flags.DEFINE_integer("filter_size_2", 5, help="Size of the filters for the second conv layer")
flags.DEFINE_integer("num_filters_2", 36, help="Number of filters in second conv layer")
flags.DEFINE_integer("fc_size", 128, help="Size of fully connected layer")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("epochs", 100, help="Number of epochs")
flags.DEFINE_float("learning_rate", 1e-4, help="Learning rate")
flags.DEFINE_float("dropout_rate", 0.2, help="Dropout rate")
flags.DEFINE_integer("ensemble_size", 5, help="Number of networks to be trained")

FLAGS = flags.FLAGS

def random_partition_test_set(data, labels, partition):
    '''
    Randomly partition a set of data and labels by a given
    percentage. This is need in this instance so that each
    network of the ensemble is trained on slightly different
    training sets.
    '''
    train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                        test_size=partition,
                                                        shuffle=True)
    train_set = (train_x, train_y)

    return train_set

def main(_):
    '''
    Main body of code for creating a two-layer convolutional neural
    network, with max pooling, flatten dense layer and softmax output.
    '''
    # Import data
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                  one_hot_encoding=True, flatten_images=False,
                  shuffle_per_epoch=True)

    for i in range(1, FLAGS.ensemble_size+1):
        # Partition the train set randomly and train the network
        print("Training model {}/{}".format(i, FLAGS.ensemble_size))
        partition = (mnist.train_y.shape[0]/FLAGS.ensemble_size)/mnist.train_y.shape[0]
        train_x, train_y = random_partition_test_set(mnist.train_x, mnist.train_y,
                                                     partition=partition)
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
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # Callbacks for saving and early stopping
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                     tf.keras.callbacks.ModelCheckpoint(filepath='best_model_nn_{}.h5'.format(i),
                                                        monitor='val_loss',
                                                        save_best_only=True)]
        model.fit(x=train_x, y=train_y,
                  epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                  callbacks=callbacks,
                  validation_data=(mnist.test_x, mnist.test_y))

if __name__ == '__main__':
    tf.app.run()

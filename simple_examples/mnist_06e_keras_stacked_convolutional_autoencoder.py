'''
Example script for constructing a 3-layer deep denoising convolutional autoencoder.
'''
from absl import flags
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dataset_utilities.mnist import MNIST

flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("epochs", 15, help="Number of epochs")
flags.DEFINE_float("learning_rate", 0.001, help="Learning rate")
flags.DEFINE_float("corruption_level", 0.4, help="Learning rate")

FLAGS = flags.FLAGS

def plotting_function(x_test, decoded_imgs):
    '''
    Simple plot for checking the input and reconstructured images.
    '''
    n_images = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        # display original
        axes = plt.subplot(2, n_images, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        # display reconstruction
        axes = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    plt.show()

def add_noise(input_data, noise_factor):
    '''
    Add noise to an array of data by a certain noise factor
    variable.
    '''
    # Add noise to the target data based on the noise factor input.
    # noise = np.random.binomial(1, 1-noise_factor, input_data.shape)
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_data.shape)
    noisy_data = input_data + noise
    # Adding noise might cause some values to surpass 0 and 1 values,
    # in which case we have to clip these.
    noisy_data = np.clip(noisy_data, 0., 1.)

    return noisy_data

def encoder(network_input):
    '''
    The encoder part of the network, using a standard CNN format
    with convolutional layers followed by max pooling.
    '''
    conv1 = tf.keras.layers.Conv2D(
        16, (3, 3), activation='relu',
        padding='same')(network_input)
    max_pool1 = tf.keras.layers.MaxPooling2D(
        (2, 2), padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(
        8, (3, 3), activation='relu',
        padding='same')(max_pool1)
    max_pool2 = tf.keras.layers.MaxPooling2D(
        (2, 2), padding='same')(conv2)
    conv3 = tf.keras.layers.Conv2D(
        8, (3, 3), activation='relu',
        padding='same')(max_pool2)
    encoded = tf.keras.layers.MaxPooling2D(
        (2, 2), padding='same')(conv3)

    return encoded

def decoder(encoder_input):
    '''
    The decoder part of the network, using a reverse structure to the
    encoder, comprising of convolutional layers and upsamling.
    '''
    conv1 = tf.keras.layers.Conv2D(
        8, (3, 3), activation='relu',
        padding='same')(encoder_input)
    upsampling_1 = tf.keras.layers.UpSampling2D(
        (2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(
        8, (3, 3), activation='relu',
        padding='same')(upsampling_1)
    upsampling_2 = tf.keras.layers.UpSampling2D(
        (2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(
        16, (3, 3), activation='relu')(upsampling_2)
    upsampling_3 = tf.keras.layers.UpSampling2D(
        (2, 2))(conv3)
    decoded = tf.keras.layers.Conv2D(
        1, (3, 3), activation='sigmoid',
        padding='same')(upsampling_3)

    return decoded

def main(_):
    '''
    Main body of code for creating a deep denoising convolutional autoencoder
    network, for performing deep reconstruction of a noisy dataset.
    '''
    # Import data
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                  one_hot_encoding=True, flatten_images=False,
                  shuffle_per_epoch=False)

    # Data information
    img_size = mnist.original_image_shape
    input_img = tf.keras.layers.Input(shape=(img_size[0], img_size[1], 1))
    encoded = encoder(input_img)
    decoded = decoder(encoded)
    # Define autoencoder input/output and optimization
    autoencoder = tf.keras.models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # Reshape the data for the network
    train_x = mnist.train_x.reshape(len(mnist.train_x), img_size[0], img_size[1], 1)
    test_x = mnist.test_x.reshape(len(mnist.test_x), img_size[0], img_size[1], 1)
    # Add noise to the data
    noisy_train_x = add_noise(train_x, FLAGS.corruption_level)
    noisy_test_x = add_noise(test_x, FLAGS.corruption_level)
    # Train the model
    autoencoder.fit(noisy_train_x, train_x,
                    epochs=FLAGS.epochs,
                    batch_size=FLAGS.batch_size,
                    shuffle=True,
                    validation_data=(test_x, test_x))

    # Retrieve the reconstructions and plot
    reconstructions = autoencoder.predict(noisy_test_x, batch_size=FLAGS.batch_size)
    plotting_function(noisy_test_x, reconstructions)

if __name__ == '__main__':
    tf.app.run()

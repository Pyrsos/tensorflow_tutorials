'''
Example script for constructing a 3-layer deep denoising autoencoder
using the Keras API.
'''
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pyrsostf.data import MNIST

flags.DEFINE_integer("autoencoder1_size", 500, help="Size of the autoencoder hidden layer")
flags.DEFINE_integer("autoencoder2_size", 200, help="Size of the autoencoder hidden layer")
flags.DEFINE_integer("autoencoder3_size", 100, help="Size of the autoencoder hidden layer")
flags.DEFINE_float("corruption_level_1", 0.5, help="Percentage of noise used to corrupt input data")
flags.DEFINE_float("corruption_level_2", 0.4, help="Percentage of noise used to corrupt input data")
flags.DEFINE_float("corruption_level_3", 0.3, help="Percentage of noise used to corrupt input data")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("pretraining_epochs", 5, help="Number of epochs")
flags.DEFINE_integer("finetuning_epochs", 15, help="Number of epochs")
flags.DEFINE_float("learning_rate", 0.001, help="Learning rate")

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
    noise = np.random.binomial(1, 1-noise_factor, input_data.shape)
    noisy_data = input_data * noise

    return noisy_data

def autoencoder_layer(input_shape, layer_size, target_data,
                      noise_rate, encoder_activation='relu',
                      decoder_activation='sigmoid', optimizer='adam',
                      loss='binary_crossentropy'):
    '''
    Build and train a single autoencoder layer using keras.
    '''
    input_data = add_noise(target_data, noise_rate)
    # Build the autoencoder
    layer_input = tf.keras.layers.Input(shape=(input_shape, ))
    encoded_input = tf.keras.layers.Input(shape=(layer_size, ))
    # Encoder and decoder transformation
    layer_encoded = tf.keras.layers.Dense(layer_size, activation=encoder_activation)
    layer_decoded = tf.keras.layers.Dense(input_shape, activation=decoder_activation)

    # Different models for the encoder and decoder sides
    layer_encoder = tf.keras.models.Model(
        inputs=layer_input, outputs=layer_encoded(layer_input))
    layer_decoder = tf.keras.models.Model(
        inputs=encoded_input, outputs=layer_decoded(encoded_input))
    autoencoder = tf.keras.models.Model(
        inputs=layer_input, outputs=layer_decoded(layer_encoded(layer_input)))
    # Training only on the decoder
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.fit(input_data, target_data,
                    epochs=FLAGS.pretraining_epochs,
                    batch_size=FLAGS.batch_size,
                    shuffle=True)
    # Retrieve the output of the encoded data
    layer_encoded_data = layer_encoder.predict(input_data,
                                               batch_size=FLAGS.batch_size)

    return layer_input, layer_encoder, layer_decoder, layer_encoded_data

def main(_):
    '''
    Main body of code for creating a deep denoising autoencoder
    network, for performing deep reconstruction of a noisy dataset.
    '''
    # Import data
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                  one_hot_encoding=True, flatten_images=True,
                  shuffle_per_epoch=True)

    # Data information
    img_size = mnist.return_input_shape()

    # First layer
    layer1_input, layer1_encoder, layer1_decoder, \
    layer1_encoded_data = autoencoder_layer(
        input_shape=img_size,
        layer_size=FLAGS.autoencoder1_size,
        target_data=mnist.train_x,
        noise_rate=FLAGS.corruption_level_1)

    # Second layer
    layer2_input, layer2_encoder, layer2_decoder, \
    layer2_encoded_data = autoencoder_layer(
        input_shape=FLAGS.autoencoder1_size,
        layer_size=FLAGS.autoencoder2_size,
        target_data=layer1_encoded_data,
        noise_rate=FLAGS.corruption_level_2)

    # Third layer
    layer3_input, layer3_encoder, layer3_decoder, \
    layer3_encoded_data = autoencoder_layer(
        input_shape=FLAGS.autoencoder2_size,
        layer_size=FLAGS.autoencoder3_size,
        target_data=layer2_encoded_data,
        noise_rate=FLAGS.corruption_level_3)

    # Deep reconstruction
    # Connect the various layers together
    reconstruction_input = layer1_input
    first_layer_encoder = layer1_encoder(reconstruction_input)
    second_layer_encoder = layer2_encoder(first_layer_encoder)
    third_layer_encoder = layer3_encoder(second_layer_encoder)
    third_layer_decoder = layer3_decoder(third_layer_encoder)
    second_layer_decoder = layer2_decoder(third_layer_decoder)
    first_layer_decoder = layer1_decoder(second_layer_decoder)

    # Define the mode with inputs and outputs
    reconstruction_model = tf.keras.models.Model(inputs=reconstruction_input,
                                                 outputs=first_layer_decoder)

    # Corrupt the input and output data
    corrupted_train_data = add_noise(mnist.train_x, FLAGS.corruption_level_1)
    corrupted_test_data = add_noise(mnist.test_x, FLAGS.corruption_level_1)

    # Optimize and fit the model
    reconstruction_model.compile(optimizer='adam',
                                 loss='binary_crossentropy')
    reconstruction_model.fit(x=corrupted_train_data, y=mnist.train_x,
                             epochs=FLAGS.finetuning_epochs, batch_size=FLAGS.batch_size,
                             shuffle=True, validation_data=(corrupted_test_data, mnist.test_x))
    # Retrieve the reconstructions and plot
    reconstructions = reconstruction_model.predict(corrupted_test_data,
                                                   batch_size=FLAGS.batch_size)
    plotting_function(corrupted_test_data, reconstructions)

if __name__ == '__main__':
    tf.app.run()

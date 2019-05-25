'''
Stacked denoising autoencoder model, built with Tensorflow and trained on
the MNIST dataset. Follows the same methodology as the Theano tutorial but
it does not perform greedy layer-wise pretraining. This is mainly due to
the graph architecture of Tensorflow that displays certain difficulties on
that end.
'''
from absl import flags
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from pyrsostf.data import MNIST
from pyrsostf.nn import AutoencoderLayer, corrupt_data_with_noise, transform_tf_variable
from pyrsostf.plots import compare_images, save_image_collection, plot_autoencoder_weights

flags.DEFINE_integer("autoencoder_size", 500, help="Size of the autoencoder hidden layer")
flags.DEFINE_float("corruption_level", 0.3, help="Percentage of noise used to corrupt input data")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("epochs", 100, help="Number of epochs")
flags.DEFINE_float("learning_rate", 0.001, help="Learning rate")

FLAGS = flags.FLAGS

def save_images(subset, feed_dict, predict_op,
                session, plot_image=True):
    '''
    Given a subset of images reconstruct these and
    save the resulting image collections.
    '''
    # Corrupt the subset with the same corruption level
    subset_noise = corrupt_data_with_noise(subset.shape, FLAGS.corruption_level)
    corrupt_images = subset * subset_noise
    # Retrieve the reconstructed images
    reconstructed_imgs = session.run(predict_op, feed_dict=feed_dict)
    # Save the image collections (both real and reconstructed)
    save_image_collection(subset, 'real')
    save_image_collection(reconstructed_imgs, 'reconstructed')

    if plot_image:
        index = np.random.randint(0, subset.shape[0])
        compare_images(subset[index], corrupt_images[index],
                       reconstructed_imgs[index])

def main(_):
    '''
    Main body of code for constructing a stacked denoising autoencoder
    model for reconstructing the MNIST dataset.
    '''
    # Load MNIST dataset
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                  one_hot_encoding=True, flatten_images=True,
                  shuffle_per_epoch=True)

    # Data information
    img_size = mnist.return_input_shape()

    # Input placeholder and corruption mask
    x_input = tf.placeholder(tf.float32, [None, img_size], name='input')

    # Build the graph
    # Encoder part
    autoencoder_1 = AutoencoderLayer(input_data=x_input,
                                     num_inputs=img_size,
                                     num_outputs=FLAGS.autoencoder_size,
                                     encoder_activation=tf.nn.relu,
                                     decoder_activation=tf.nn.sigmoid)
    autoencoder_2 = AutoencoderLayer(input_data=autoencoder_1.encoded_input,
                                     num_inputs=FLAGS.autoencoder_size,
                                     num_outputs=512,
                                     encoder_activation=tf.nn.relu,
                                     decoder_activation=tf.nn.softplus)
    autoencoder_3 = AutoencoderLayer(input_data=autoencoder_2.encoded_input,
                                     num_inputs=512,
                                     num_outputs=256,
                                     encoder_activation=tf.nn.relu,
                                     decoder_activation=tf.nn.softplus)
    # Latent space
    latent_space = autoencoder_3.encoded_input
    # Reconstructed input
    layer_3_reconstruction = autoencoder_3.decoder(latent_space)
    layer_2_reconstruction = autoencoder_2.decoder(layer_3_reconstruction)
    layer_1_reconstruction = autoencoder_1.decoder(layer_2_reconstruction)
    input_reconstruction = layer_1_reconstruction
    # Cost and optimization
    cost = tf.reduce_sum(tf.abs(input_reconstruction - x_input))
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)
    predict_op = autoencoder_1.decoded_input

    # Initialize tensorflow session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    loss = 0

    with tqdm(total=FLAGS.epochs, postfix='Loss = {:.3f}'.format(loss)) as epoch_progress:
        # Iterate through epochs
        for epoch in range(FLAGS.epochs):
            with tqdm(total=len(mnist), postfix='Loss: {:.3f}'.format(loss), mininterval=1e-4,
                      leave=True) as batch_progress:
                # Train model per batch
                for batch, (batch_x, _) in enumerate(mnist):
                    batch_noise_1 = corrupt_data_with_noise(batch_x.shape, FLAGS.corruption_level)
                    batch_noise_2 = corrupt_data_with_noise([batch_x.shape[0],
                                                             autoencoder_1.num_outputs],
                                                            FLAGS.corruption_level)
                    batch_noise_3 = corrupt_data_with_noise([batch_x.shape[0],
                                                             autoencoder_2.num_outputs],
                                                            FLAGS.corruption_level)
                    feed_dict_train = {x_input:batch_x,
                                       autoencoder_1.mask:batch_noise_1,
                                       autoencoder_2.mask:batch_noise_2,
                                       autoencoder_3.mask:batch_noise_3}
                    _, loss = session.run([train_op, cost], feed_dict=feed_dict_train)
                    # Update bar
                    batch_progress.set_postfix(Loss=loss)
                    batch_progress.update()

            # Noise for the different layers
            test_set_noise_layer_1 = corrupt_data_with_noise(mnist.test_x.shape,
                                                             FLAGS.corruption_level)
            test_set_noise_layer_2 = corrupt_data_with_noise([mnist.test_x.shape[0],
                                                              autoencoder_1.num_outputs],
                                                             FLAGS.corruption_level)
            test_set_noise_layer_3 = corrupt_data_with_noise([mnist.test_x.shape[0],
                                                              autoencoder_2.num_outputs],
                                                             FLAGS.corruption_level)
            feed_dict_test = {x_input: mnist.test_x,
                              autoencoder_1.mask: test_set_noise_layer_1,
                              autoencoder_2.mask: test_set_noise_layer_2,
                              autoencoder_3.mask: test_set_noise_layer_3}
            test_loss = session.run(cost, feed_dict=feed_dict_test)
            # Update bar
            epoch_progress.set_postfix(Loss=test_loss)
            epoch_progress.update()

    weights = transform_tf_variable(session, autoencoder_1.weights)
    plot_autoencoder_weights(weights)
    # Define a subset of the test set for reconstruction visualization.
    subset = mnist.test_x[:100]
    # Noise for the different layers
    subset_layer_1 = corrupt_data_with_noise(subset.shape,
                                             FLAGS.corruption_level)
    subset_layer_2 = corrupt_data_with_noise([subset.shape[0], autoencoder_1.num_outputs],
                                             FLAGS.corruption_level)
    subset_layer_3 = corrupt_data_with_noise([subset.shape[0], autoencoder_2.num_outputs],
                                             FLAGS.corruption_level)
    feed_dict_vis = {x_input: subset,
                     autoencoder_1.mask: subset_layer_1,
                     autoencoder_2.mask: subset_layer_2,
                     autoencoder_3.mask: subset_layer_3}
    save_images(subset, feed_dict_vis, predict_op, session)

if __name__ == '__main__':
    tf.app.run()

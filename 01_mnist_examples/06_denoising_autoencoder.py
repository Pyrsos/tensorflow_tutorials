from absl import flags
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dataset_utilities.mnist import MNIST
from utilities.tf_utilities import new_weights, new_biases, AutoencoderLayer, corrupt_data_with_noise
from utilities.plot_utilities import compare_images, save_image_collection

flags.DEFINE_integer("autoencoder_size", 500, help="Size of the autoencoder hidden layer")
flags.DEFINE_float("corruption_level", 0.3, help="Percentage of noise used to corrupt input data")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
flags.DEFINE_integer("epochs", 100, help="Number of epochs")
flags.DEFINE_float("learning_rate", 0.01, help="Learning rate")

FLAGS = flags.FLAGS

def save_images(subset, x, mask, predict_op,
                session, plot_image=True):
    # Corrupt the subset with the same corruption level
    subset_noise = corrupt_data_with_noise(subset, FLAGS.corruption_level)
    corrupt_images = subset * subset_noise
    # Create the feed dict to pass to the graph
    feed_dict = {x:subset, mask: subset_noise}
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
    # Load MNIST dataset
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                    one_hot_encoding=True, flatten_images=True,
                    shuffle_per_epoch=True)

    # Data information
    img_size = mnist.return_input_shape()
    corruption_level = FLAGS.corruption_level

    # Input placeholder and corruption mask
    x = tf.placeholder(tf.float32, [None, img_size], name='input')
    mask = tf.placeholder(tf.float32, [None, img_size], name='mask')

    # Build the graph
    ae = AutoencoderLayer(input_data=x, mask=mask,
                        num_inputs=img_size,
                        num_outputs=FLAGS.autoencoder_size)

    # Cost and optimization
    cost = tf.reduce_sum(tf.pow(x-ae.decoded_input, 2))
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)
    predict_op = ae.decoded_input

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    loss = 0

    with tqdm(total=FLAGS.epochs, postfix='Loss = {:.3f}'.format(loss)) as epoch_progress:
        # Iterate through epochs
        for epoch in range(FLAGS.epochs):
            with tqdm(total=len(mnist), postfix='Loss: {:.3f}'.format(loss), mininterval=1e-4,
                      leave=True) as batch_progress:
                # Train model per batch
                for batch, (batch_x, batch_y) in enumerate(mnist):
                    batch_noise = corrupt_data_with_noise(batch_x, FLAGS.corruption_level)
                    feed_dict_train = {x:batch_x, mask:batch_noise}
                    _, loss = session.run([train_op, cost], feed_dict=feed_dict_train)
                    # Update bar
                    batch_progress.set_postfix(Loss=loss)
                    batch_progress.update()

            test_set_noise = corrupt_data_with_noise(mnist.test_x, FLAGS.corruption_level)
            feed_dict_test = {x: mnist.test_x, mask: test_set_noise}
            test_cost = session.run(cost, feed_dict=feed_dict_test)
            # Update bar
            epoch_progress.set_postfix(Loss=loss)
            epoch_progress.update()

    subset = mnist.test_x[:100]
    save_images(subset, x, mask, predict_op, session)

if __name__ == '__main__':
    tf.app.run()

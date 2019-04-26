'''
Example script for loading a pre-trained model
for classifying the MNIST dataset using keras.
'''
from absl import flags
import numpy as np
import tensorflow as tf
from dataset_utilities.mnist import MNIST
from utilities.plot_utilities import (plot_conv_weights, plot_conv_layer, print_confusion_matrix,
                                      find_wrong_predictions, plot_images)

flags.DEFINE_string("model_path", "model.keras", help="Path to the model file")
flags.DEFINE_integer("batch_size", 100, help="Batch size")
FLAGS = flags.FLAGS

def main(_):
    '''
    Main body of code for loading a pre-trained model and accessing
    various functionalities of the system.
    '''
    # Import data
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                  one_hot_encoding=True, flatten_images=False,
                  shuffle_per_epoch=True)
    img_size = mnist.original_image_shape
    num_classes = mnist.return_num_classes()
    # Load model
    model = tf.keras.models.load_model('best_model.h5')
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
    # Get model summary
    model.summary()
    model.evaluate(x=mnist.test_x, y=mnist.test_y)
    # Get the input layer
    input_layer = model.layers[0]
    # Retrieve convolutional layers and weights
    conv1_layer = model.layers[2]
    conv1_weights = conv1_layer.get_weights()[0]
    conv2_layer = model.layers[5]
    conv2_weights = conv2_layer.get_weights()[0]
    # Plot convolutional weights
    plot_conv_weights(conv1_weights)
    plot_conv_weights(conv2_weights)
    # Output of convolutions
    conv1_output = tf.keras.backend.function(
        inputs=[input_layer.input], outputs=[conv1_layer.output])
    conv2_output = tf.keras.backend.function(
        inputs=[input_layer.input], outputs=[conv2_layer.output])
    # Get an example image to plot the convolutions
    image_instance = np.array([mnist.test_x[0]])
    conv_1_ex_output = conv1_output(image_instance)[0]
    conv_2_ex_output = conv2_output(image_instance)[0]
    plot_conv_layer(conv_1_ex_output)
    plot_conv_layer(conv_2_ex_output)

if __name__ == '__main__':
    tf.app.run()

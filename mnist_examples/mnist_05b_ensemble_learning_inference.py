'''
Example script for loading an ensemble of CNNs
for classifying the MNIST dataset using keras.
'''
import os
from absl import flags
import numpy as np
import tensorflow as tf
from dataset_utilities.mnist import MNIST
from utilities.plot_utilities import print_confusion_matrix, find_wrong_predictions, plot_images

flags.DEFINE_string("ensemble_networks_path", "ensemble_network_models", help="Path to the model file")
flags.DEFINE_integer("batch_size", 100, help="Batch size")

FLAGS = flags.FLAGS

def main(_):
    '''
    Main body of code for loading the pre-trained models and accessing
    various functionalities of the system.
    '''
    # Import data
    mnist = MNIST(batch_size=FLAGS.batch_size, normalize_data=True,
                  one_hot_encoding=True, flatten_images=False,
                  shuffle_per_epoch=True)
    img_size = mnist.original_image_shape
    num_classes = mnist.return_num_classes()

    # Find number of models in folder
    model_files = os.listdir(FLAGS.ensemble_networks_path)
    num_models = len(model_files)
    # Lists to append the models and the logits
    ensemble_models = []
    ensemble_logits = []

    for i in range(1, num_models+1):
        print("Loading model {}/{}".format(i, num_models))
        # Load model and get the performance
        model = tf.keras.models.load_model(os.path.join(
            FLAGS.ensemble_networks_path, 'ensemble_network_{}.h5'.format(i)))
        model.evaluate(x=mnist.test_x,
                       y=mnist.test_y,
                       batch_size=FLAGS.batch_size)
        logits = model.predict(x=mnist.test_x)
        ensemble_logits.append(logits)
        predictions = np.argmax(logits, axis=1)

    # Get the mean of the logits from the models and
    # their predictions
    ensemble_logits = np.array(ensemble_logits)
    mean_logits = np.mean(ensemble_logits, axis=0)
    predictions = np.argmax(mean_logits, axis=1)

    # Plot the confusion matrix
    print_confusion_matrix(labels=mnist.test_y_cls,
                           predictions=predictions,
                           num_classes=num_classes)
    # Get images that are classified incorrectly
    wrong_indeces, wrong_images, wrong_labels, correct_labels = find_wrong_predictions(labels=mnist.test_y_cls,
                                                                                       predictions=predictions,
                                                                                       images=mnist.test_x)
    incorrect_logits = mean_logits[wrong_indeces]
    plot_images(images=wrong_images[:5], y_pred=incorrect_logits[:5],
                logits=incorrect_logits[:5], cls_true=correct_labels[:5],
                cls_pred=wrong_labels[:5], img_shape=img_size)
    # Get the accuracy of the ensemble, to compare to the accuracy
    # of the individual networks.
    print("Accuracy of ensemble network: {}".format(1 - (wrong_images.shape[0]/mnist.test_x.shape[0])))

if __name__ == '__main__':
    tf.app.run()

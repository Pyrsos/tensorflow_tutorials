import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def find_wrong_predictions(labels, predictions, images):
    '''
    Find the instances of the set where the system has made a wrong
    prediction and return both the actual image and the wrong prediction.
    '''
    wrong_predictions = np.where(labels != predictions)[0]

    wrong_images = images[wrong_predictions]
    wrong_labels = predictions[wrong_predictions]
    correct_labels = labels[wrong_predictions]

    return wrong_images, wrong_labels, correct_labels

def plot_images(images, cls_true, img_shape, cls_pred=None):
    '''
    Plot some of the images from the dataset.
    '''
    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, axes in enumerate(axes.flat):
        # Plot image
        axes.imshow(images[i].reshape(img_shape), cmap='binary')
        # Show true and predicted classes
        if cls_pred is None:
            xlabel = "True: {}".format(cls_true[i])
        else:
            xlabel = "True: {}, Pred: {}".format(cls_true[i], cls_pred[i])

        axes.set_xlabel(xlabel)

        # Remove ticks from the plot
        axes.set_xticks([])
        axes.set_yticks([])

    plt.show()

def print_confusion_matrix(labels, predictions, num_classes):
    '''
    Given the correct labels, predictions and number of classes
    print the confusion matrix to evaluate the system's performance.
    '''
    # Get the confusion_matrix
    conf_mat = confusion_matrix(y_true=labels,
                                y_pred=predictions)
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    # Plot adjustments
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.xlabel('True')

    plt.show()

def plot_weights(weights, img_shape):
    '''
    Plot the weights of the network.
    '''
    w_min = np.min(weights)
    w_max = np.max(weights)

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, axes in enumerate(axes.flat):
        # There are 12 sublplots but we only have 10 digits
        if i < 10:
            # Get the weights for each digit (i) and reshape them
            # to match the original image shape (28, 28) instead of
            # flattened image shape (784)
            image = weights[:, i].reshape(img_shape)
            # Set subplot label
            axes.set_xlabel("Weights: {}".format(i))
            # Plot image
            axes.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        axes.set_xticks([])
        axes.set_yticks([])

    plt.show()

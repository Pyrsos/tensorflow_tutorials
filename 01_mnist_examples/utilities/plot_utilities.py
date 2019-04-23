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

def plot_images(images, y_pred, logits, cls_true, cls_pred, img_shape):
    '''
    Plot some of the images from the dataset.
    '''
    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(images.shape[0], 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    bar_colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'b', 'r', 'g']
    # First print the images
    for i, subfig in enumerate(axes[:, 0]):
        # Plot image
        subfig.imshow(images[i].reshape(img_shape), cmap='binary', aspect='auto')
        # Show true and predicted classes
        xlabel = "True: {}, Pred: {}".format(cls_true[i], cls_pred[i])
        subfig.set_xlabel(xlabel)
        # Remove ticks from the plot
        subfig.set_xticks([])
        subfig.set_yticks([])
    # Then print the logits
    for i, subfig in enumerate(axes[:, 1]):
        subfig.bar(np.arange(10), y_pred[i], color=bar_colors)
        subfig.set_xticks(np.arange(10))
        subfig.set_yticks([])

    for i, subfig in enumerate(axes[:, 2]):
        subfig.bar(np.arange(10), logits[i], color=bar_colors)
        subfig.set_xticks(np.arange(10))
        subfig.set_yticks([])

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

def plot_conv_weights(weights, input_channel=0):
    '''
    Plot the weights of the convolutional layers.
    '''
    # Get min and max for the weights
    # to adjust any issues with colors.
    w_min = np.min(weights)
    w_max = np.max(weights)
    # Get the number of filters used in the conv layer.
    num_filters = weights.shape[3]
    # Num of grids to plot (rounded up to the square root
    # of the number of filters.
    num_grids = int(np.ceil(np.sqrt(num_filters)))
    # Figure with a grid of sub-plots
    fig, axes = plt.subplots(num_grids, num_grids)
    # Plot all the filter-weights
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = weights[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def plot_conv_layer(layer_output):
    '''
    Plot the output of a convolutional layer.
    '''
    num_filters = layer_output.shape[3]
    num_grids = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = layer_output[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

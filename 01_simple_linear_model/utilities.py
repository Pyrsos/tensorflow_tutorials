import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_images(images, cls_true, img_shape, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        # Show true and predicted classes
        if cls_pred is None:
            xlabel = "True: {}".format(cls_true[i])
        else:
            xlabel = "True: {}, Pred: {}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def print_confusion_matrix(labels, predictions, num_classes):
    # Get the confusion_matrix
    cm = confusion_matrix(y_true=labels,
                          y_pred=predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
    w_min = np.min(weights)
    w_max = np.max(weights)

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # There are 12 sublplots but we only have 10 digits
        if i < 10:
            # Get the weights for each digit (i) and reshape them
            # to match the original image shape (28, 28) instead of
            # flattened image shape (784)
            image = weights[:, i].reshape(img_shape)
            # Set subplot label
            ax.set_xlabel("Weights: {}".format(i))
            # Plot image
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

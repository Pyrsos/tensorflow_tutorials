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

def print_confusion_matrix(data, predictions):
    # Define the true classes and number of classes
    cls_true = data.y_test_cls
    num_classes = data.num_classes
    # Get the predicted classes
    cls_pred = predictions
    # Get the confusion_matrix
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
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

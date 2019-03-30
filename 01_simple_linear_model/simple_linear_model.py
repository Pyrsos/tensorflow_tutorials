import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from utilities import plot_images
from mnist import MNIST

# Load MNIST dataset

data = MNIST(data_dir="data/MNIST/")

# Print some information regarding the mnist

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t{}".format(data.num_test))

# Tuple with image dimensions
img_size_flat = data.img_size_flat
# Size of image when flatten (28 x 28 = 784)
img_shape = data.img_shape
# Number of classes
num_classes = data.num_classes


import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from utilities import plot_images
# Load MNIST dataset from Keras

train, test = tf.keras.datasets.mnist.load_data()

train_x, train_y_cls = train
test_x, test_y_cls = test
# Split the train set to train and validation

valid_x, valid_y_cls = train_x[55000:], train_y_cls[55000:]
train_x, train_y_cls = train[:55000], train_y_cls[:55000]

# Transform labels to one-hot encoding

train_y = tf.keras.utils.to_categorical(train_y_cls)
valid_y = tf.keras.utils.to_categorical(valid_y_cls)
test_y = tf.keras.utils.to_categorical(test_y_cls)

# Print some information regarding the mnist

print("Size of:")
print("- Training-set:\t\t{}".format(train_y_cls.shape))
print("- Validation-set:\t{}".format(valid_y_cls.shape))
print("- Test-set:\t{}".format(test_y_cls.shape))

# Define some of the dimensions

sample_image = train[0][0]

# Tuple with image dimensions
img_shape = sample_image.shape
# Size of image when flatten (28 x 28 = 784)
img_size_flat = sample_image.shape[0] * sample_image.shape[1]
# Number of classes
num_classes = np.unique(train[1])


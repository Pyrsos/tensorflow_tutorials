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

# Define tensorflow graph.

# First define the input placeholder for features
# (in this case images) and labels (one-hot for y_true
# and labels y_true_cls).

x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

# This is a simple implementation of logistic regression
# so we can easily define our own weights and biases.

weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# Define the output of the computation. In this case it will
# simply be the (input x weights) +  bias .

logits = tf.matmul(x, weights) + biases

# Then we need to activate our logits, and this is done by
# applying a softmax function over them.

y_pred = tf.nn.softmax(logits)

# As the predictions are in one-hot encoding, we can transform
# these easily using an argmax function, to get the predicted label.

y_pred_cls = tf.argmax(y_pred, axis=1)

# Define cost and optimizations methods.

# First we need to define the cost function that
# we want to minimize. In this case we will use the cross entropy.

cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=logits)
cost = tf.reduce_mean(cross_entropy)

# Then we define an optmization method. In this case we use standard
# gradient descent.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# Define performance metrics to measure the model's response
# In this case we will just check the accuracy of the classification
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run the graph
session = tf.Session()
session.run(tf.global_variables_initializer())

# Training loop
num_iterations = 10
batch_size = 1000
for i in range(num_iterations):
    # Train
    x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)
    feed_dict_train = {x: x_batch,
                 y_true: y_true_batch}
    _ = session.run(optimizer, feed_dict=feed_dict_train)
    # Test
    feed_dict_test = {x:data.x_test,
                      y_true: data.y_test,
                      y_true_cls:data.y_test_cls}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test set: {0:.1%}".format(acc))

import tensorflow as tf
from utilities import plot_images, print_confusion_matrix
from mnist import MNIST

class LogisticRegression():
    '''
    Logistic Regression class.
    '''
    def __init__(self, data, batch_size, num_iterations,
                 input_size, output_layer_size):
        self._data = data
        self._batch_size = batch_size
        self._num_iterations = num_iterations
        self._input_size = input_size
        self._output_layer_size = output_layer_size

        self._x, self._y_true, self._y_true_cls = self.__model_input()
        self._weights, self._biases, self._logits = self.__init_weights_and_bias()
        self._y_pred, self._y_pred_cls = self.__model_output()
        self._cost = self.__cost_calculation()
        self._accuracy = self.__accuracy_calculation()

        self._optimizer = tf.train.AdamOptimizer(learning_rate=0.5).minimize(self._cost)

        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

    def __model_input(self):
        '''
        Tensorflow input graph definition.
        '''
        x = tf.placeholder(tf.float32, [None, self._input_size])
        y_true = tf.placeholder(tf.float32, [None, self._output_layer_size])
        y_true_cls = tf.placeholder(tf.int64, [None])

        return x, y_true, y_true_cls

    def __init_weights_and_bias(self):
        '''
        Initialise the weights and bias matrices and calculate the logits
        '''
        weights = tf.Variable(tf.zeros([self._input_size, self._output_layer_size]))
        biases = tf.Variable(tf.zeros([self._output_layer_size]))
        logits = tf.matmul(self._x, weights) + biases

        return weights, biases, logits

    def __model_output(self):
        '''
        Logits activation and class prediction.
        '''
        y_pred = tf.nn.softmax(self._logits)
        y_pred_cls = tf.argmax(y_pred, axis=1)

        return y_pred, y_pred_cls

    def __cost_calculation(self):
        '''
        Calculate the loss of the system and use to further optimize.
        '''
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self._y_true, logits=self._logits)
        cost = tf.reduce_mean(cross_entropy)

        return cost

    def __accuracy_calculation(self):
        '''
        Calculate the accuracy of the system to gauge its performance.
        '''
        correct_prediction = tf.equal(self._y_pred_cls, self._y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def train(self):
        for i in range(self._num_iterations):
            batch_x, batch_y, _ = self._data.random_batch(batch_size=self._batch_size)
            feed_dict_train = {self._x: batch_x, self._y_true: batch_y}
            _ = self._session.run(self._optimizer, feed_dict=feed_dict_train)
            feed_dict_test = {self._x: self._data.x_test,
                              self._y_true: self._data.y_test,
                              self._y_true_cls: self._data.y_test_cls}

            acc = self._session.run(self._accuracy, feed_dict=feed_dict_test)
            print("Accuracy on test set: {0:.1%}".format(acc))

def main():

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

    # Call model
    model = LogisticRegression(data=data, batch_size=128, num_iterations=6000,
                               input_size=img_size_flat, output_layer_size=num_classes)
    model.train()

if __name__ == '__main__':
    main()


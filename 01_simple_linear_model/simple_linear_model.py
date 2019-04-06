import tensorflow as tf
from utilities import plot_images, print_confusion_matrix, plot_weights
from mnist import MNIST

class LogisticRegression():
    '''
    Logistic Regression class.
    '''
    def __init__(self, batch_size, num_iterations,
                 input_size, output_layer_size):
        self._batch_size = batch_size
        self._num_iterations = num_iterations
        self._input_size = input_size
        self._output_layer_size = output_layer_size

        self._x, self._y_true, self._y_true_cls = self.__model_input()
        self._weights, self._biases, self._logits = self.__init_weights_and_bias()
        self._y_pred, self._y_pred_cls = self.__model_output()
        self._cost = self.__cost_calculation()
        self._accuracy = self.__accuracy_calculation()

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(self._cost)

        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

    def __model_input(self):
        '''
        Tensorflow input graph definition.
        '''
        features_x = tf.placeholder(tf.float32, [None, self._input_size])
        y_true = tf.placeholder(tf.float32, [None, self._output_layer_size])
        y_true_cls = tf.placeholder(tf.int64, [None])

        return features_x, y_true, y_true_cls

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
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self._y_true,
                                                        logits=self._logits)
        cost = tf.reduce_mean(cross_entropy)

        return cost

    def __accuracy_calculation(self):
        '''
        Calculate the accuracy of the system to gauge its performance.
        '''
        correct_prediction = tf.equal(self._y_pred_cls, self._y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def return_predictions(self, test_x, test_y, test_y_cls):
        '''
        Return predictions for the test set.
        '''
        feed_dict_test = {self._x: test_x,
                          self._y_true: test_y,
                          self._y_true_cls: test_y_cls}
        predictions = self._session.run(self._y_pred_cls, feed_dict_test)

        return predictions

    def return_weights(self):
        '''
        Return the weights in numpy matrix in order to visualize.
        '''
        weights = self._session.run(self._weights)

        return weights

    def train_step(self, batch_x, batch_y):
        '''
        Single train step, given a batch of features and labels.
        A loop can then be used to iterate over this function multiple
        times to train the system.
        '''
        feed_dict_train = {self._x: batch_x, self._y_true: batch_y}
        _, loss = self._session.run([self._optimizer, self._cost], feed_dict=feed_dict_train)

        return loss

    def validation_cycle(self, test_x, test_y, test_y_cls):
        '''
        Feed the entire set to the system in order to validate the
        response. This is not optimal as the validation set should be
        ideally passed in batches. 
        '''
        feed_dict_test = {self._x: test_x,
                          self._y_true: test_y,
                          self._y_true_cls: test_y_cls}

        acc = self._session.run(self._accuracy, feed_dict=feed_dict_test)

        return acc

def main():
    '''
    Main function to execute. Loading MNIST and training model.
    '''

    # Load MNIST dataset
    mnist = MNIST(batch_size=100, normalize_data=True,
                  one_hot_encoding=True, flatten_images=True,
                  shuffle_per_epoch=True)

    epochs = 100
    img_size_flat = mnist.return_input_shape()
    num_classes = mnist.return_num_classes() 

    # Call model
    model = LogisticRegression(batch_size=100, num_iterations=1000,
                               input_size=img_size_flat, output_layer_size=num_classes)


    # Main train loop to iterate over epochs and then
    # over batches. Once per epoch we validate the system's
    # accuracy on the test set.
    for i in range(1, epochs+1):
        for j, (batch_x, batch_y) in enumerate(mnist):
            loss = model.train_step(batch_x, batch_y)
            print("Epoch {}/{}, Batch: {}/{} with Loss: {}".format(
                i, epochs, j, len(mnist), loss))
        acc = model.validation_cycle(mnist.test_x, mnist.test_y, mnist.test_y_cls)
        print("Epoch {}/{}, Accuracy: {}".format(i, epochs, acc))

if __name__ == '__main__':
    main()

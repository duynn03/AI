import matplotlib.pyplot as plt
import numpy as np

np.random.seed(100)


# https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/

# The class that represents our network's hidden and output layers.
class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    # init n_input, n_neuron, activation, weight, bias of layer
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """

        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        self.activation = activation
        self.bias = bias if bias is not None else np.random.rand(n_neurons)

    def activate(self, x):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result after active.
        """

        z = np.dot(x, self.weights) + self.bias
        self.a = self.apply_activation(z)
        return self.a

    def apply_activation(self, z):
        """
        Applies the chosen activation function (if any).
        :param z: The normal value.
        :return: The activated value.
        """

        # In case no activation function was chosen
        if self.activation is None:
            return z

        # tanh
        if self.activation == 'tanh':
            return np.tanh(z)

        # sigmoid
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))

        return z

    def apply_activation_derivative(self, z):
        """
        Applies the derivative of the activation function (if any).
        :param z: The normal value.
        :return: The "derived" value.
        """

        # We use 'z' directly here because its already activated, the only values that
        # are used in this function are the a that were saved.

        if self.activation is None:
            return z

        if self.activation == 'tanh':
            return 1 - z ** 2

        if self.activation == 'sigmoid':
            return z * (1 - z)

        return z


class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self.layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        for layer in self.layers:
            X = layer.activate(X)

        return X

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        y_hat = self.feed_forward(X)

        # Loop over the layers backward
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # If this is the output layer
            if layer == self.layers[-1]:
                layer.error = y - y_hat

                # The y_hat = layer.z in this case
                layer.delta = layer.error * layer.apply_activation_derivative(y_hat)
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.a)

        # Update the weights
        for i in range(len(self.layers)):
            layer = self.layers[i]

            # The input is either the previous layers output or X itself (for the first hidden layer)
            input = np.atleast_2d(X if i == 0 else self.layers[i - 1].a)

            # update weights
            layer.weights += layer.delta * input.T * learning_rate

    """
    N.B: Having a sigmoid activation in the output layer can be interpreted
    as expecting probabilities as outputs.
    W'll need to choose a winning class, this is usually done by choosing the
    index of the biggest probability.
    """

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)

        # One row
        if ff.ndim == 1:
            return np.argmax(ff)

        # Multiple rows
        return np.argmax(ff, axis=1)

    def train(self, X, y, learning_rate, max_epochs):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """

        errors = []

        # sử dụng SGD
        for i in range(max_epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)

            # At every 10th epoch, we will print out the Mean Squared Error and save it in mses which we will return at the end.
            if i % 10 == 0:
                mse = np.mean(np.square(y - nn.feed_forward(X)))
                errors.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        return errors

    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """

        return (y_pred == y_true).mean()


# init the network
nn = NeuralNetwork()
nn.add_layer(Layer(2, 3, 'tanh'))
nn.add_layer(Layer(3, 3, 'sigmoid'))
nn.add_layer(Layer(3, 2, 'sigmoid'))

# Define dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Train the neural network
errors = nn.train(X, y, 0.3, 290)
print('Accuracy: %.2f%%' % (nn.accuracy(nn.predict(X), y.flatten()) * 100))

# Plot changes in mse
plt.plot(errors)
plt.title('Changes in MSE')
plt.xlabel('Epoch (every 10th)')
plt.ylabel('MSE')
plt.show()

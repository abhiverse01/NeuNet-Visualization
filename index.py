import numpy as np

class NeuralNetwork:
    """
    A simple feedforward neural network implemented from scratch using NumPy.

    Attributes:
        input_size (int): The number of input neurons.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output neurons.
        weights_input_hidden (numpy.ndarray): Weights connecting input to hidden layer.
        weights_hidden_output (numpy.ndarray): Weights connecting hidden to output layer.
        bias_hidden (numpy.ndarray): Biases for the hidden layer.
        bias_output (numpy.ndarray): Biases for the output layer.

    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network.

        Args:
            input_size (int): The number of input neurons.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases using random values
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Sigmoid activation of the input.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Derivative of the sigmoid activation of the input.
        """
        return x * (1 - x)

    def forward(self, X):
        """
        Performs a forward pass through the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the network.
        """
        # Calculate hidden layer activations
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Calculate output layer activations
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_layer_input)

        return self.output

    def backward(self, X, y, learning_rate):
        """
        Performs backpropagation to update weights and biases.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target output.
            learning_rate (float): Learning rate for weight updates.
        """
        # Calculate output layer error
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """
        Trains the neural network.

        Args:
            X (numpy.ndarray): Training input data.
            y (numpy.ndarray): Training target output.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight updates.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass and weight updates
            self.backward(X, y, learning_rate)

            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(np.square(y - output))}")

# Example usage:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Test the trained network
print("Predictions:")
print(nn.forward(X))
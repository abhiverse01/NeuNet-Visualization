import numpy as np

class NeuralNetwork:
    """
    A flexible feedforward neural network implemented from scratch using NumPy.
    """

    def __init__(self, layer_sizes, activation='sigmoid', loss='mse', optimizer='sgd', regularization=None, reg_lambda=0.01):
        """
        Initializes the neural network with the specified architecture.

        Args:
            layer_sizes (list): A list of integers where each element represents the number of neurons in that layer.
            activation (str): The activation function to use ('sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax').
            loss (str): The loss function to use ('mse', 'cross_entropy').
            optimizer (str): The optimization algorithm to use ('sgd', 'adam').
            regularization (str): The regularization technique to use ('l2', 'dropout').
            reg_lambda (float): The regularization parameter (lambda) for L2 regularization.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = self.get_activation_function(activation)
        self.activation_derivative = self.get_activation_derivative(activation)
        self.loss_function = self.get_loss_function(loss)
        self.loss_derivative = self.get_loss_derivative(loss)
        self.optimizer = optimizer
        self.regularization = regularization
        self.reg_lambda = reg_lambda

        # Initialize weights and biases using random values
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.num_layers - 1)]

        # Initialize optimizer parameters
        if optimizer == 'adam':
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0

    def get_activation_function(self, name):
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'tanh':
            return np.tanh
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'leaky_relu':
            return lambda x: np.where(x > 0, x, x * 0.01)
        elif name == 'softmax':
            return lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function")

    def get_activation_derivative(self, name):
        if name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif name == 'tanh':
            return lambda x: 1 - np.tanh(x)**2
        elif name == 'relu':
            return lambda x: np.where(x > 0, 1, 0)
        elif name == 'leaky_relu':
            return lambda x: np.where(x > 0, 1, 0.01)
        elif name == 'softmax':
            return lambda x: x * (1 - x)
        else:
            raise ValueError("Unsupported activation function")

    def get_loss_function(self, name):
        if name == 'mse':
            return lambda y_true, y_pred: np.mean(np.square(y_true - y_pred))
        elif name == 'cross_entropy':
            return lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-8))
        else:
            raise ValueError("Unsupported loss function")

    def get_loss_derivative(self, name):
        if name == 'mse':
            return lambda y_true, y_pred: y_pred - y_true
        elif name == 'cross_entropy':
            return lambda y_true, y_pred: y_pred - y_true
        else:
            raise ValueError("Unsupported loss function")

    def forward(self, X):
        """
        Performs a forward pass through the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            list: Outputs of all layers, including the input layer.
        """
        activations = [X]
        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z)
            activations.append(a)
        return activations

    def backward(self, activations, y, learning_rate):
        """
        Performs backpropagation to update weights and biases.

        Args:
            activations (list): Outputs of all layers, including the input layer.
            y (numpy.ndarray): Target output.
            learning_rate (float): Learning rate for weight updates.
        """
        output_error = self.loss_derivative(y, activations[-1])
        deltas = [output_error * self.activation_derivative(activations[-1])]

        for i in range(self.num_layers - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.activation_derivative(activations[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(self.num_layers - 1):
            if self.optimizer == 'sgd':
                self.weights[i] -= activations[i].T.dot(deltas[i]) * learning_rate
                self.biases[i] -= np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            elif self.optimizer == 'adam':
                self.t += 1
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * activations[i].T.dot(deltas[i])
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * np.square(activations[i].T.dot(deltas[i]))
                m_hat_weights = self.m_weights[i] / (1 - self.beta1**self.t)
                v_hat_weights = self.v_weights[i] / (1 - self.beta2**self.t)
                self.weights[i] -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * np.sum(deltas[i], axis=0, keepdims=True)
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * np.square(np.sum(deltas[i], axis=0, keepdims=True))
                m_hat_biases = self.m_biases[i] / (1 - self.beta1**self.t)
                v_hat_biases = self.v_biases[i] / (1 - self.beta2**self.t)
                self.biases[i] -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

    def train(self, X, y, epochs, learning_rate, batch_size=None, validation_data=None, early_stopping=False, patience=5, verbose=True):
        """
        Trains the neural network.

        Args:
            X (numpy.ndarray): Training input data.
            y (numpy.ndarray): Training target output.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight updates.
            batch_size (int): Size of each mini-batch.
            validation_data (tuple): Validation data (X_val, y_val) for early stopping.
            early_stopping (bool): If True, stop training early if validation loss doesn't improve.
            patience (int): Number of epochs to wait for improvement before stopping.
            verbose (bool): If True, print progress during training.
        """
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            if batch_size:
                permutation = np.random.permutation(X.shape[0])
                X_shuffled = X[permutation]
                y_shuffled = y[permutation]
                for i in range(0, X.shape[0], batch_size):
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]
                    activations = self.forward(X_batch)
                    self.backward(activations, y_batch, learning_rate)
            else:
                activations = self.forward(X)
                self.backward(activations, y, learning_rate)

            if verbose and (epoch + 1) % 100 == 0:
                loss = self.loss_function(y, self.forward(X)[-1])
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

            if validation_data:
                X_val, y_val = validation_data
                val_loss = self.loss_function(y_val, self.forward(X_val)[-1])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if early_stopping and patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                    break

    def predict(self, X):
        """
        Predicts the output for the given input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output predictions.
        """
        return self.forward(X)[-1]

    def save_model(self, file_path):
        """
        Saves the model weights and biases to a file.

        Args:
            file_path (str): The path to the file where the model will be saved.
        """
        model = {'weights': self.weights, 'biases': self.biases}
        np.save(file_path, model)

    def load_model(self, file_path):
        """
        Loads the model weights and biases from a file.

        Args:
            file_path (str): The path to the file from which the model will be loaded.
        """
        model = np.load(file_path, allow_pickle=True).item()
        self.weights = model['weights']
        self.biases = model['biases']

# Example usage:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(layer_sizes=[2, 4, 1], activation='sigmoid', optimizer='adam')
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Test the trained network
print("Predictions:")
print(nn.predict(X))

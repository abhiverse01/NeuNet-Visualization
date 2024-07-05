<body>
    <h1>Neural_Nets_Scratch</h1>
    <h2>Explanation: Creating an Entire Neural Network from Scratch using numpy and math.</h2>
    <h3>NeuralNetwork Class:</h3>
    <ol>
        <li>
            <strong>__init__:</strong> Initializes the network with input, hidden, and output layer sizes. It also initializes weights and biases randomly.
        </li>
        <li>
            <strong>sigmoid:</strong> Implements the sigmoid activation function, which introduces non-linearity into the network.
        </li>
        <li>
            <strong>sigmoid_derivative:</strong> Calculates the derivative of the sigmoid function, used in backpropagation.
        </li>
        <li>
            <strong>forward:</strong> Performs a forward pass through the network, calculating activations at each layer.
        </li>
        <li>
            <strong>backward:</strong> Implements backpropagation to calculate gradients and update weights and biases.
        </li>
        <li>
            <strong>train:</strong> Trains the network for a specified number of epochs, performing forward and backward passes in each epoch.
        </li>
    </ol>
    <h2>Example Usage:</h2>
    <ul>
        <li>Creates sample input (X) and target output (y) data.</li>
        <li>Initializes a NeuralNetwork instance with desired layer sizes.</li>
        <li>Trains the network using the train method.</li>
        <li>Tests the trained network by making predictions on the input data.</li>
    </ul>
    <h2>Key Concepts:</h2>
    <ol>
        <li><strong>Feedforward Network:</strong> Information flows in one direction, from input to output, without loops.</li>
        <li><strong>Activation Function:</strong> Introduces non-linearity, allowing the network to learn complex patterns.</li>
        <li><strong>Weights:</strong> Parameters that determine the strength of connections between neurons.</li>
        <li><strong>Biases:</strong> Constants added to neuron activations, influencing their firing thresholds.</li>
        <li><strong>Backpropagation:</strong> Algorithm for calculating gradients and updating weights based on the error between predicted and actual outputs.</li>
    </ol>
    <h2>Requirements:</h2>
    <pre>
        <code>numpy==1.21.0</code>
    </pre>
    <h2>Installation:</h2>
    <p>Clone the repository:</p>
    <pre>
        <code>git clone https://github.com/yourusername/Neural_Nets_Scratch.git</code>
    </pre>
    <p>Navigate to the project directory:</p>
    <pre>
        <code>cd Neural_Nets_Scratch</code>
    </pre>
    <p>Install the required dependencies:</p>
    <pre>
        <code>pip install -r requirements.txt</code>
    </pre>
    <h2>License:</h2>
    <pre>
        <code>MIT License</code>
    </pre>
</body>

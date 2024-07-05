# Neural_Nets_Scratch

## Explanation: Creating an Entire Neural Network from Scratch using numpy and math.

### NeuralNetwork Class:
1. __init__: Initializes the network with input, hidden, and output layer sizes. It also initializes weights and biases randomly.
2. sigmoid: Implements the sigmoid activation function, which introduces non-linearity into the network.
sigmoid_derivative: Calculates the derivative of the sigmoid function, used in backpropagation.
3. forward: Performs a forward pass through the network, calculating activations at each layer.
4. backward: Implements backpropagation to calculate gradients and update weights and biases.
5. train: Trains the network for a specified number of epochs, performing forward and backward passes in each epoch.

## Example Usage:
- Creates sample input (X) and target output (y) data.
- Initializes a NeuralNetwork instance with desired layer sizes.
- Trains the network using the train method.
- Tests the trained network by making predictions on the input data.

## Key Concepts:

1. Feedforward Network: Information flows in one direction, from input to output, without loops.
2. Activation Function: Introduces non-linearity, allowing the network to learn complex patterns.
3. Weights: Parameters that determine the strength of connections between neurons.
4. Biases: Constants added to neuron activations, influencing their firing thresholds.
5. Backpropagation: Algorithm for calculating gradients and updating weights based on the error between predicted and actual outputs.
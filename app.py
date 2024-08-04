# necessary libraries and imports
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import Flask, jsonify, request
from index import NeuralNetwork  # Assuming the neural network code is in index.py


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Initialize neural network
nn = None

@app.route('/init', methods=['POST'])
def init_network():
    global nn
    data = request.json
    layer_sizes = data.get('layer_sizes', [3, 5, 2])
    activation = data.get('activation', 'sigmoid')
    loss = data.get('loss', 'mse')
    optimizer = data.get('optimizer', 'sgd')
    
    nn = NeuralNetwork(layer_sizes, activation, loss, optimizer)
    return jsonify({"message": "Neural Network Initialized"}), 200

@app.route('/train', methods=['POST'])
def train_network():
    global nn
    data = request.json
    X = np.array(data['X'])
    y = np.array(data['y'])
    epochs = data.get('epochs', 100)
    learning_rate = data.get('learning_rate', 0.01)
    
    if nn is None:
        return jsonify({"error": "Neural Network not initialized"}), 400
    
    losses = nn.train(X, y, epochs=epochs, learning_rate=learning_rate)
    return jsonify({"message": "Training Complete", "losses": losses}), 200

@app.route('/state', methods=['GET'])
def get_network_state():
    global nn
    if nn is None:
        return jsonify({"error": "Neural Network not initialized"}), 400
    
    state = {
        "weights": [w.tolist() for w in nn.weights],
        "biases": [b.tolist() for b in nn.biases],
    }
    return jsonify(state), 200

if __name__ == '__main__':
    app.run(debug=True)

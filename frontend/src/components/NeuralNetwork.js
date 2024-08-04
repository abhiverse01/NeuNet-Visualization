// src/components/NeuralNetwork.js
import React, { useState } from 'react';
import Header from './Header';
import InitializeNetwork from './InitializeNetwork';
import TrainNetwork from './TrainNetwork';
import Losses from './Losses';

const NeuralNetwork = () => {
    const [message, setMessage] = useState('');
    const [losses, setLosses] = useState([]);

    const initializeNetwork = async () => {
        const response = await fetch('http://127.0.0.1:5000/init', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                layer_sizes: [3, 5, 2],
                activation: 'relu',
                loss: 'mse',
                optimizer: 'sgd'
            }),
        });

        const data = await response.json();
        setMessage(data.message);
    };

    const trainNetwork = async () => {
        const response = await fetch('http://127.0.0.1:5000/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                X: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                y: [[0], [1]],
                epochs: 100,
                learning_rate: 0.01
            }),
        });

        const data = await response.json();
        setMessage(data.message);
        setLosses(data.losses || []);
    };

    return (
        <div>
            <Header />
            <InitializeNetwork onInitialize={initializeNetwork} message={message} />
            <TrainNetwork onTrain={trainNetwork} message={message} />
            <Losses losses={losses} />
        </div>
    );
};

export default NeuralNetwork;

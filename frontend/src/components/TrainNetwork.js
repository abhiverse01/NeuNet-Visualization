// src/components/TrainNetwork.js
import React from 'react';

const TrainNetwork = ({ onTrain, message }) => {
    return (
        <div>
            <h2>Train Neural Network</h2>
            <button onClick={onTrain}>Train Network</button>
            <p>{message}</p>
        </div>
    );
};

export default TrainNetwork;

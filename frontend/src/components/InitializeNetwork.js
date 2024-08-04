// src/components/InitializeNetwork.js
import React from 'react';

const InitializeNetwork = ({ onInitialize, message }) => {
    return (
        <div>
            <h2>Initialize Neural Network</h2>
            <button onClick={onInitialize}>Initialize Network</button>
            <p>{message}</p>
        </div>
    );
};

export default InitializeNetwork;

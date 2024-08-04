// src/components/Losses.js
import React from 'react';

const Losses = ({ losses }) => {
    return (
        <div>
            <h2>Training Losses</h2>
            <ul>
                {losses && losses.map((loss, index) => (
                    <li key={index}>Epoch {index + 1}: {loss}</li>
                ))}
            </ul>
        </div>
    );
};

export default Losses;

import React, { useState } from 'react';
import { predictRating } from '../services/api';

const PredictionForm = () => {
    const [formData, setFormData] = useState({
        year: '',
        genre: '',
        language: '',
        duration: ''
    });
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const rating = await predictRating(formData);
            setPrediction(rating);
            setError(null);
        } catch (err) {
            setError('Error getting prediction');
            setPrediction(null);
        }
    };

    return (
        <div>
            <h2>Movie Rating Predictor</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Year:</label>
                    <input
                        type="number"
                        name="year"
                        value={formData.year}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Genre:</label>
                    <input
                        type="text"
                        name="genre"
                        value={formData.genre}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Language:</label>
                    <input
                        type="text"
                        name="language"
                        value={formData.language}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Duration (minutes):</label>
                    <input
                        type="number"
                        name="duration"
                        value={formData.duration}
                        onChange={handleChange}
                        required
                    />
                </div>
                <button type="submit">Predict Rating</button>
            </form>
            {prediction && (
                <div>
                    <h3>Predicted Rating: {prediction.toFixed(2)}</h3>
                </div>
            )}
            {error && (
                <div style={{ color: 'red' }}>
                    {error}
                </div>
            )}
        </div>
    );
};

export default PredictionForm;
import React, { useState, useEffect } from 'react';
import { predictRating, trainModel, deleteModel, getModelStatus } from './api';
import './App.css';

function App() {
	const [formData, setFormData] = useState({
		year: '',
		genre: '',
		language: '',
		duration: ''
	});
	const [prediction, setPrediction] = useState(null);
	const [error, setError] = useState(null);
	const [modelStatus, setModelStatus] = useState({ trained: false, message: 'Checking model status...' });
	const [isTraining, setIsTraining] = useState(false);

	// Check model status on component mount
	useEffect(() => {
		checkModelStatus();
	}, []);

	const checkModelStatus = async () => {
		try {
			const status = await getModelStatus();
			setModelStatus(status);
		} catch (err) {
			setError('Error checking model status');
		}
	};

	const handleChange = (e) => {
		setFormData({
			...formData,
			[e.target.name]: e.target.value
		});
	};

	const handleSubmit = async (e) => {
		e.preventDefault();
		try {
			// Clean and validate the data before sending
			const cleanedData = {
				year: parseInt(formData.year),
				duration: parseInt(formData.duration),
				genre: formData.genre.trim(),
				language: formData.language.trim()
			};

			// Validate data
			if (!cleanedData.year || cleanedData.year < 1900 || cleanedData.year > new Date().getFullYear()) {
				throw new Error('Please enter a valid year between 1900 and ' + new Date().getFullYear());
			}
			if (!cleanedData.duration || cleanedData.duration < 1) {
				throw new Error('Please enter a valid duration (minutes)');
			}
			if (!cleanedData.genre) {
				throw new Error('Please enter a genre');
			}
			if (!cleanedData.language) {
				throw new Error('Please enter a language');
			}

			const result = await predictRating(cleanedData);
			if (result && typeof result.predicted_rating === 'number') {
				setPrediction(result.predicted_rating);
				setError(null);
			} else {
				throw new Error('Invalid response from server');
			}
		} catch (err) {
			setError(err.message || 'Error getting prediction');
			setPrediction(null);
		}
	};

	const handleModelAction = async () => {
		try {
			setIsTraining(true);
			setError(null);

			if (modelStatus.trained) {
				// Delete the model
				await deleteModel();
			} else {
				// Train the model
				await trainModel();
			}

			// Force a small delay to ensure the backend has updated
			await new Promise(resolve => setTimeout(resolve, 500));
			const newStatus = await getModelStatus();
			console.log('New model status:', newStatus);
			setModelStatus(newStatus);
		} catch (err) {
			console.error('Model action error:', err);
			setError(err.message || `Error ${modelStatus.trained ? 'deleting' : 'training'} model`);
		} finally {
			setIsTraining(false);
		}
	};

	return (
		<div className="App">
			<header className="App-header">
				<h1>Movie Rating Prediction</h1>
			</header>
			<main className="App-main">
				<div className="model-status">
					<p>
						<strong>Model status: </strong>
						{isTraining ?
							(modelStatus.trained ? 'Deleting model...' : 'Training model...')
							: modelStatus.message}
					</p>
					<button
						className={`model-button ${modelStatus.trained ? 'delete' : 'train'}`}
						onClick={handleModelAction}
						disabled={isTraining}
					>
						{isTraining ?
							(modelStatus.trained ? 'Deleting...' : 'Training...')
							: (modelStatus.trained ? 'Delete Model' : 'Train Model')}
					</button>
				</div>
				<div className="prediction-form">
					<form onSubmit={handleSubmit}>
						<div className="form-group">
							<label>Year:</label>
							<input
								type="number"
								name="year"
								value={formData.year}
								onChange={handleChange}
								required
							/>
						</div>
						<div className="form-group">
							<label>Genre:</label>
							<input
								type="text"
								name="genre"
								value={formData.genre}
								onChange={handleChange}
								required
							/>
						</div>
						<div className="form-group">
							<label>Language:</label>
							<input
								type="text"
								name="language"
								value={formData.language}
								onChange={handleChange}
								required
							/>
						</div>
						<div className="form-group">
							<label>Duration (minutes):</label>
							<input
								type="number"
								name="duration"
								value={formData.duration}
								onChange={handleChange}
								required
							/>
						</div>
						<button type="submit" className="submit-button">Predict Rating</button>
					</form>
					{prediction && (
						<div className="prediction-result">
							<h3>Predicted Rating: {prediction.toFixed(2)}</h3>
						</div>
					)}
					{error && (
						<div className="error-message">
							{error}
						</div>
					)}
				</div>
			</main>
		</div>
	);
}

export default App;
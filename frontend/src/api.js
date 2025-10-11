const API_BASE_URL = 'http://localhost:5001';

export const predictRating = async (movieData) => {
	try {
		console.log('Sending prediction request with data:', movieData);

		const response = await fetch(`${API_BASE_URL}/api/predict`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(movieData)
		});

		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.error || 'Failed to get prediction');
		}

		console.log('Received prediction response:', data);
		return data;
	} catch (error) {
		console.error('Prediction error:', error);
		throw new Error(error.message || 'Error making prediction request');
	}
};

export const trainModel = async () => {
	try {
		const response = await fetch(`${API_BASE_URL}/api/train`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			}
		});

		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.error || 'Failed to train model');
		}

		return data;
	} catch (error) {
		console.error('Training error:', error);
		throw new Error(error.message || 'Error training model');
	}
};

export const deleteModel = async () => {
	try {
		const response = await fetch(`${API_BASE_URL}/api/delete`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			}
		});

		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.error || 'Failed to delete model');
		}

		return data;
	} catch (error) {
		console.error('Delete error:', error);
		throw new Error(error.message || 'Error deleting model');
	}
};

export const getModelStatus = async () => {
	try {
		const response = await fetch(`${API_BASE_URL}/api/status`);
		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.error || 'Failed to get model status');
		}

		return data;
	} catch (error) {
		console.error('Status check error:', error);
		throw new Error(error.message || 'Error checking model status');
	}
};
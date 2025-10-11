#!/bin/bash

# Exit on error
set -e

echo "Setting up Movie Rating Prediction App..."

# Setup Python packages globally
echo "Setting up Python backend..."
cd backend
pip3 install -r requirements.txt

# Setup React frontend
echo "Setting up React frontend..."
cd ../frontend
npm install

echo "Setup complete! You can now start the application:"
echo "1. Start the backend server:"
echo "   cd backend"
echo "   python app.py"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   cd frontend"
echo "   npm start"
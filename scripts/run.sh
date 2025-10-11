#!/bin/bash

# Exit on error
set -e

echo "Starting Movie Rating Prediction App..."

# Function to cleanup background processes on exit
cleanup() {
    echo "Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start backend
echo "Starting Python backend..."
cd "$(dirname "$0")/../backend"
python3 api/app.py &

# Start frontend
echo "Starting React frontend..."
cd ../frontend
npm start &

# Keep the script running
echo "Services are running. Press Ctrl+C to stop..."
wait
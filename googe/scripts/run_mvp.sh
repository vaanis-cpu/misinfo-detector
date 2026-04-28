#!/bin/bash
# Run MVP stack locally without Docker

set -e

echo "Starting Misinfo Detector MVP..."

# Check Python version
python_version=$(python3 --version 2>&1 || python --version 2>&1 || echo "unknown")
echo "Python version: $python_version"

# Create data directory
mkdir -p data models/cache models/checkpoints

# Install dependencies if not already installed
if ! pip show fastapi > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Seed data
echo "Seeding database..."
python scripts/seed_data.py

# Start API server
echo "Starting API server on http://localhost:8000"
cd "$(dirname "$0")/.."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
sleep 3

# Start dashboard
echo "Starting dashboard on http://localhost:8501"
streamlit run src/dashboard/app.py --server.port 8501 &
DASH_PID=$!

echo ""
echo "========================================"
echo "Misinfo Detector MVP is running!"
echo "========================================"
echo "API:       http://localhost:8000"
echo "Dashboard: http://localhost:8501"
echo "API Docs:  http://localhost:8000/docs"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop"

# Wait for interrupt
trap "kill $API_PID $DASH_PID 2>/dev/null; exit" INT TERM
wait

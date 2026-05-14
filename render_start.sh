#!/bin/bash
set -e

echo "=== ZeRO Stock Forecast — Render Startup ==="
echo "Checking model files..."

# Only download if models directory is empty or missing
if [ ! -d "models/joblib" ] || [ -z "$(ls -A models/joblib 2>/dev/null)" ]; then
    echo "Models not found — downloading from model-store branch..."

    # Install git if not present (Render images have it but just in case)
    which git || apt-get install -y git

    # Download only the zip file from the model-store branch
    # using GitHub's raw content API — no full clone needed
    curl -L \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3.raw" \
        "https://api.github.com/repos/glitching-gops/Agentic-Stock-Forecast/contents/models_store.zip?ref=model-store" \
        -o models_store.zip

    echo "Download complete. Extracting..."
    python -c "
import zipfile
import os
with zipfile.ZipFile('models_store.zip', 'r') as zf:
    zf.extractall('.')
print('Extraction complete.')
"
    rm models_store.zip
    echo "Models ready."

    # Count extracted files
    joblib_count=$(ls models/joblib/*.joblib 2>/dev/null | wc -l)
    lstm_count=$(ls models/lstm/*.pt 2>/dev/null | wc -l)
    meta_count=$(ls models/meta/*.joblib 2>/dev/null | wc -l)
    echo "Extracted: $joblib_count XGBoost, $lstm_count LSTM, $meta_count meta models"
else
    echo "Models already present — skipping download."
    joblib_count=$(ls models/joblib/*.joblib 2>/dev/null | wc -l)
    echo "Found: $joblib_count XGBoost models"
fi

echo ""
echo "Initialising database..."
python -c "from data.db import init_db; init_db(); print('DB init OK')"

echo ""
echo "Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT

#!/bin/bash
set -e

echo "=== ZeRO Stock Forecast — Render Startup ==="
echo "Port: $PORT"
echo "Checking model files..."

# Ensure models directory exists
mkdir -p models/joblib models/lstm models/meta

# Only download if models directory is empty
if [ ! -f "models/joblib/RELIANCE.NS.joblib" ]; then
    echo "Models not found — downloading from model-store branch..."

    if [ -z "$GITHUB_TOKEN" ]; then
        echo "ERROR: GITHUB_TOKEN is not set. Cannot download models."
        exit 1
    fi

    # Download zip
    curl -L \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3.raw" \
        "https://api.github.com/repos/glitching-gops/Agentic-Stock-Forecast/contents/models_store.zip?ref=model-store" \
        -o models_store.zip

    echo "Download complete. Size: $(du -h models_store.zip | cut -f1). Extracting..."
    python3 -c "
import zipfile
import os
print('Starting extraction...')
with zipfile.ZipFile('models_store.zip', 'r') as zf:
    zf.extractall('.')
print('Extraction complete.')
"
    rm models_store.zip
    echo "Models ready."
else
    echo "Models already present — skipping download."
fi

echo "Initialising database..."
# Use a timeout for DB init to prevent hanging
timeout 30s python3 -c "from data.db import init_db; init_db(); print('DB init OK')" || echo "DB init timed out or failed (ignoring to allow startup)"

echo "Starting FastAPI server on port $PORT..."
# exec ensures uvicorn receives signals directly from Render
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info

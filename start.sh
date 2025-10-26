#!/usr/bin/env bash
set -e # exit immediately on errors

echo "SmartTwin Setup and Launch Script"

# --- 1. Check dependencies and environment ---
if ! command -v uv &>/dev/null; then
  echo "uv not found, installing via pip..."
  pip install uv
fi

# --- 2. Sync environment (create .venv if missing) ---
echo "Syncing environment..."
uv sync

# --- 3. Activate environment ---
source .venv/bin/activate

# --- 4. Initial model training (if model file missing) ---
MODEL_PATH="data/waste_model.pkl"
if [ ! -f "$MODEL_PATH" ]; then
  echo "No existing model found. Running initial training..."
  uv run python trainning/daily_train_predict_waste.py
else
  echo "✅ Existing model detected at $MODEL_PATH"
fi

# --- 5. Launch Streamlit app ---
echo "Starting SmartTwin dashboard..."
streamlit run app.py

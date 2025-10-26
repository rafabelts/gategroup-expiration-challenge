#!/usr/bin/env bash
set -e

echo "SmartTwin Setup and Launch Script"

if ! command -v python3 &>/dev/null; then
  echo "Python3 not found. Please install Python 3.9+ first."
  exit 1
fi

if ! command -v pip &>/dev/null; then
  echo "pip not found. Attempting to install with ensurepip..."
  python3 -m ensurepip --upgrade || {
    echo "Failed to install pip. Please install it manually."
    exit 1
  }
fi

if ! command -v uv &>/dev/null; then
  echo "uv not found, installing via pip..."
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install uv
fi

echo "Syncing environment..."
uv sync

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment activation script not found."
  exit 1
fi

MODEL_PATH="data/waste_model.pkl"
if [ ! -f "$MODEL_PATH" ]; then
  echo "No existing model found. Running initial training..."
  uv run python trainning/daily_train_predict_waste.py
else
  echo "Existing model detected at $MODEL_PATH"
fi

echo "Starting SmartTwin dashboard..."
uv run streamlit run app.py

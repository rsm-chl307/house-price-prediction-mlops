#!/bin/bash

# Exit immediately if a command fails
set -e

echo "------------------------------------------"
echo "Starting Streamlit UI..."
echo "------------------------------------------"

streamlit run scripts/ui.py

echo "UI stopped."
#!/bin/bash
# This script installs all required packages from requirements.txt for Colab.

echo "Installing all required packages..."
pip install -q -r requirements.txt
echo "Installation complete. You may need to restart the runtime."

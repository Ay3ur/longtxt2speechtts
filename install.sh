#!/bin/bash

# Install Ubuntu Updates
sudo apt update && sudo apt upgrade -y

# Install python3-venv
sudo apt install python3-venv -y

# Create a virtual environment
python3 -m venv env

#Grant execution permissions ton run.sh
chmod +x run.sh

# Activate the virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install ffmpeg
sudo apt install ffmpeg -y

# Run the Streamlit application
streamlit run app.py


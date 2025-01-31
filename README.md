# Keystroke Dynamics Recognition System

## Overview
This repository contains a keystroke dynamics recognition system designed to collect and analyze typing behavior for user identification. It includes scripts for data collection, subject renaming, training a machine learning model, and real-time recognition.

## Folder Structure
```
├── data_collection.py    # Script for keystroke data collection
├── change_name_sub.py    # Script for renaming subjects in collected data
├── model.py              # Training script for user identification
├── emotions.py           # Model for emotion recognition based on typing patterns
├── recognition.py        # Real-time keystroke authentication script
├── graphs/               # Visualization scripts for keystroke data analysis
└── data/                 # Folder storing collected keystroke data
```


## Usage
### 1. Collecting Data
Run the `data_collection.py` script to collect keystroke data for user identification:
```sh
python data_collection.py
```
This script records the user's typing behavior when entering a predefined password and saves the data into structured CSV files.

### 2. Renaming Subject IDs
After collecting data, run `change_name_sub.py` to ensure compatibility across multiple data files:
```sh
python change_name_sub.py
```
This script assigns unique subject IDs to each dataset.

### 3. Training the Model
To train the user recognition model, execute:
```sh
python model.py
```
This script trains an LSTM-based deep learning model using the collected dataset and a benchmark dataset from [Kaggle](https://www.kaggle.com/datasets/carnegiecylab/keystroke-dynamics-benchmark-data-set).

### 4. Emotion Recognition
To analyze emotional states based on typing patterns, run:
```sh
python emotions.py
```
This script applies machine learning techniques to classify typing behavior into emotional states.

### 5. Real-Time User Recognition
To perform real-time keystroke authentication, execute:
```sh
python recognition.py
```
This script listens to the keyboard, extracts keystroke dynamics, and identifies the user.

### 6. Updating the Model with New Data
If new data is collected, retrain the model using:
```sh
python model.py
```

## Data Visualization
The `graphs/` folder contains scripts for analyzing and visualizing keystroke data. These tools help in interpreting the collected dataset and model performance.

## Dataset
The model is trained using:
- **Collected user data** (from `data_collection.py`)
- **Benchmark dataset**: [Keystroke Dynamics Benchmark Dataset](https://www.kaggle.com/datasets/carnegiecylab/keystroke-dynamics-benchmark-data-set)

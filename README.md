# Wine Classification with MLflow

This project trains and tracks a machine learning model for wine classification using **Python**, **scikit-learn**, and **MLflow**.

The goal is to classify wine samples based on their chemical characteristics and use MLflow to organize experiments, parameters, metrics, and trained models.

## Project Overview

This repository demonstrates a simple machine learning workflow:

1. Load the wine dataset
2. Prepare the data for training
3. Train a classification model
4. Evaluate model performance
5. Track experiments with MLflow
6. Save the trained model

## Technologies Used

* Python
* scikit-learn
* pandas
* MLflow
* NumPy

## Project Structure

```text
wine-mlflow-classification/
├── data/                  # Dataset files, if applicable
├── notebooks/             # Exploratory notebooks, if applicable
├── src/                   # Source code
├── mlruns/                # MLflow experiment tracking files
├── requirements.txt       # Project dependencies
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Hudson-Borgo/wine-mlflow-classification.git
cd wine-mlflow-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training script

```bash
python src/train.py
```

### 5. Open MLflow UI

```bash
mlflow ui
```

Then open the MLflow dashboard in your browser:

```text
http://localhost:5000
```

## MLflow Tracking

MLflow is used to track:

* Model parameters
* Evaluation metrics
* Experiment runs
* Trained model artifacts

This makes it easier to compare different models and reproduce experiments.

## Model Evaluation

The model is evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score

## Purpose

This project was created for learning and practicing:

* Machine learning classification
* Experiment tracking
* MLflow workflows
* Basic MLOps concepts

## Author

Hudson Borgo

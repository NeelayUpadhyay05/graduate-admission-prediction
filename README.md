# Graduate Admission Prediction Using ANN
***
## Objective

Build an Artificial Neural Network (ANN) to predict the probability of graduate school admission based on applicant profiles.

## Introduction

Graduate admissions are influenced by several quantitative and qualitative factors, such as standardized test scores, academic record, and research experience. Accurately estimating an applicant’s admission chance helps institutions screen candidates and guides applicants on their admission prospects.

## Abstract

This project uses an open-source dataset containing profiles of applicants to graduate programs in the US. Key features include GRE Score, TOEFL Score, University Rating, Statement of Purpose (SOP), Letter of Recommendation (LOR), CGPA, and Research experience. A regression-based ANN is trained to predict `Chance of Admit`, a probability between 0 and 1.

## Purpose

- Predict admission probability to help guide students and automate part of the screening process.
- Demonstrate end-to-end regression modeling with neural networks on tabular data.
- Showcase preprocessing, normalization, model training, and evaluation in a real-world ML workflow.


## Dataset

- Source: [Kaggle - Graduate Admission 2](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv)
- 500 samples, 7 features (+1 target):
    - GRE Score (out of 340)
    - TOEFL Score (out of 120)
    - University Rating (1–5)
    - SOP strength (1–5)
    - LOR strength (1–5)
    - CGPA (out of 10)
    - Research (0 or 1)
- Target: Chance of Admit (float, 0.0–1.0)


## Project Workflow

- **Data Loading \& Cleaning:**
    - Import CSV, remove unneeded columns (`Serial No.`)
    - Check for duplicates and missing values
- **Feature Preparation:**
    - Features (`X`): All fields except `Chance of Admit`
    - Target (`y`): `Chance of Admit`
    - Split into train/test sets (80/20)
    - Scale features with MinMaxScaler to [0-1]
- **Model Architecture:**
    - Keras Sequential ANN
        - **Input:** 7 features
        - **Layers:** 3 layers (Dense(14, relu), Dense(14, relu), Dense(1, linear))
        - **Nodes per layer:** 14, 14, 1
    - Compiled with mean squared error loss and Adam optimizer
- **Training \& Evaluation:**
    - **Epochs:** 100
    - Train with validation split
    - Monitor training and validation loss
    - Predict and evaluate test set with R² (coefficient of determination)
- **Visualization:**
    - Plot training and validation loss curves


## Results

- **Sample Performance:**
    - R² score on test set: ~0.80
    - RMSE/loss tracked during training
- **Visualization:**
    - Loss curve shows convergence and possible overfitting/underfitting


## How to Run

1. Place `admission_predict_ver1.1.csv` in the working directory.
2. Run `graduate_admission_prediction.ipynb` in Jupyter or Colab.
3. Follow the in-notebook cells—dataset is loaded, preprocessed, model is trained and evaluated.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow / keras
- matplotlib


## License

Open for academic/research use.

## Acknowledgements

- Kaggle for dataset
- TensorFlow/Keras docs for ANN guidance

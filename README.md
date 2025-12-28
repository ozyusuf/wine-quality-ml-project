# Wine Quality & Type Prediction Project

This project is a Machine Learning application that aims to make two fundamental predictions based on the chemical properties of wines.

## 🎯 Project Objectives

Two different machine learning models have been developed within the scope of this project:

1.  **Score Prediction Model**:
    *   **Goal**: To predict the tasting score of the wine (between 0-10).
    *   **Method**: Regression.
    *   **Motivation**: Grading wine quality with a precise score instead of just "Good/Bad" provides more detailed feedback to producers.
    *   **Architecture**: Uses **Random Forest Regressor** or **XGBoost Regressor** to predict a continuous variable from chemical features.

2.  **Type Prediction Model**:
    *   **Goal**: To predict whether the wine is Red or White.
    *   **Method**: Binary Classification.
    *   **Motivation**: To analyze the distinctiveness of chemical components on wine type and to detect labeling errors.
    *   **Architecture**: Uses **Logistic Regression** or **Random Forest Classifier**.

## � Data & Preprocessing

The dataset is derived from the **Wine Quality Dataset** (UCI Machine Learning Repository).

**Preprocessing Steps:**
1.  **Duplicate Removal**: Identical rows are removed to differeniate strictly between samples.
2.  **Missing Value Imputation**: `SimpleImputer` (median strategy) is used for any missing numeric values.
3.  **Scaling**: `StandardScaler` normalizes features to have mean=0 and variance=1, which is crucial for model performance (especially for XGBoost and Regression).
4.  **Outlier Removal**: Interquartile Range (IQR) method is applied to training data to reduce noise.
5.  **Stratified Split**: Ensures the training and test sets have the same proportion of classes (Red/White or Good/Bad).

## �🚀 Installation

To install the necessary libraries:

```bash
pip install -r requirements.txt
```

## 📊 Metrics and Performance

### 1. Score Prediction (Regression)
*   **MSE (Mean Squared Error)**: The average of the squared errors.
*   **R2 Score**: How well the model explains the variance in the data.
*   **Visualizations**: **Actual vs Predicted** plot.

### 2. Type Prediction (Classification)
*   **Accuracy**: The ratio of correctly predicted observations.
*   **Precision & Recall**: The model's ability to find relevant cases and be precise.
*   **Confusion Matrix**: A table showing correct and incorrect predictions for each class.

## 📂 File Structure

*   `combined_analysis.py`: The main script for analysis and modeling.
*   `src/`: Contains utility functions and data processing modules.
*   `combined_outputs/`: Stores model outputs, metrics, and visualization plots.

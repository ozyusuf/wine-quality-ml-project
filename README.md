# Wine Quality Machine Learning Project

## 1. Project Overview & Problem Definition

This project aims to develop a robust, production-ready Machine Learning system for wine analysis. It has been expanded to address three distinct predictive challenges:

### 1.1. Quality Classification (Original Goal)
*   **Objective:** Classify wines into "Premium" (Quality $\ge$ 7) or "Standard" (Quality < 7).
*   **Type:** Binary Classification.
*   **Business Value:** Automate quality assurance and identifying high-value batches.

### 1.2. Score Prediction (NEW)
*   **Objective:** Predict the exact quality score of a wine on a scale of 0-10.
*   **Type:** Regression.
*   **Motivation:** While binary classification is useful for filtering, a precise score provides finer granularity for grading and pricing.
*   **Architecture:** **Random Forest Regressor**. Evaluation of a non-linear regression model to capture complex chemical interactions that determine a specific score.

### 1.3. Type Prediction (NEW)
*   **Objective:** Predict whether a wine is **Red** or **White** based on its chemical properties.
*   **Type:** Binary Classification.
*   **Motivation:** To test the hypothesis that chemical signatures are distinct enough to perfectly segregate wine types without visual inspection.
*   **Architecture:** **XGBoost Classifier**. A Gradient Boosting decision tree model chosen for its high accuracy and ability to handle tabular chemical data effectively.

---

## 2. Requirements

The project relies on the following key Python libraries:
*   **pandas**: Data manipulation and analysis.
*   **numpy**: Numerical computing.
*   **matplotlib & seaborn**: Data visualization.
*   **scikit-learn**: Machine learning models (Random Forest) and metrics.
*   **xgboost**: Advanced Gradient Boosting models.
*   **shap**: Explainable AI (Model interpretation).

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## 3. Metrics & Visualization Plan

### 3.1. Score Prediction (Regression)
To evaluate the 0-10 score predictor:
*   **Metrics:**
    *   **MSE (Mean Squared Error):** Measures average squared difference between estimated values and the actual value.
    *   **RMSE (Root Mean Squared Error):** Square root of MSE, interpretable in the original units (0-10 score).
    *   **R² Score:** The proportion of the variance in the dependent variable that is predictable from the independent variables.
*   **Visualizations:**
    *   **Actual vs. Predicted Plot:** Scatter plot to visualize correlation.
    *   **Residual Plot:** To check for heteroscedasticity or bias.
    *   **Feature Importance:** Which chemicals drive the score?

### 3.2. Type Prediction (Classification)
To evaluate the Red vs. White classifier:
*   **Metrics:**
    *   **Accuracy:** Overall correctness.
    *   **Precision & Recall:** To ensure no class is ignored.
    *   **F1-Score:** Harmonic mean of precision and recall.
*   **Visualizations:**
    *   **Confusion Matrix:** To see exactly how many Reds were mistaken for Whites.
    *   **ROC Curve (Receiver Operating Characteristic):** To illustrate diagnostic ability.
    *   **Feature Importance:** Which chemicals distinguish Red from White?

---

## 4. Usage

### Run Quality Classification (Original)
```bash
python combined_analysis.py
```

### Run Score Prediction (New)
```bash
python score_prediction.py
```
*Outputs will be saved in `outputs/score_prediction/`*

### Run Type Prediction (New)
```bash
python type_prediction.py
```
*Outputs will be saved in `outputs/type_prediction/`*

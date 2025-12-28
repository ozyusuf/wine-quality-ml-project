import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

# Add project root to path to verify imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom utility functions
from src.utils import (load_data_for_score, load_data_for_type, evaluate_model,
                       plot_actual_vs_predicted, plot_residuals, plot_confusion_matrix,
                       plot_roc_curve, plot_correlation_matrix, plot_learning_curve,
                       plot_xgboost_loss_curve)

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# Configuration Paths
RED_PATH = os.path.join(os.path.dirname(__file__), 'red_wine', 'red_wine_cleaned.csv')
WHITE_PATH = os.path.join(os.path.dirname(__file__), 'white_wine', 'white_wine_cleaned.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'combined_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_score_prediction():
    """
    Executes the Score Prediction workflow (Regression task).
    Predicts wine quality score (0-10) based on chemical features.
    """
    print("\n" + "="*50)
    print("### TASK 1: SCORE PREDICTION (REGRESSION) ###")
    print("="*50)
    
    # 1. Load Data
    X, y = load_data_for_score(RED_PATH, WHITE_PATH)
    if X is None: return

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model 1: Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    rf_reg = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_reg.fit(X_train, y_train)
    rf_metrics, rf_pred, _ = evaluate_model(rf_reg, X_test, y_test, "RF Regressor", task_type='regression')
    
    # 4. Model 2: XGBoost Regressor
    print("\nTraining XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    # Fit with eval_set to track loss over iterations
    xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    xgb_metrics, xgb_pred, _ = evaluate_model(xgb_reg, X_test, y_test, "XGB Regressor", task_type='regression')
    
    # 5. Visualizations (Using the Best Model)
    # XGBoost is typically chosen for detailed plotting here
    best_pred = xgb_pred
    best_model_name = "XGBoost"
    
    print("\nGenerating Regression Plots...")
    plot_actual_vs_predicted(y_test, best_pred, os.path.join(OUTPUT_DIR, 'score_actual_vs_pred.png'), title=f"{best_model_name}: Actual vs Predicted Score")
    plot_residuals(y_test, best_pred, os.path.join(OUTPUT_DIR, 'score_residuals.png'), title=f"{best_model_name}: Residuals")
    
    # Learning Curve (New for GitHub Requirements)
    print("Generating Learning Curve...")
    plot_learning_curve(xgb_reg, X_train, y_train, os.path.join(OUTPUT_DIR, 'score_learning_curve.png'), 
                        title=f"Learning Curve ({best_model_name})", scoring='r2', ylabel='R2 Score')

    # Loss Curve (New for GitHub Requirements - "Loss ve Accuracy eğrileri")
    print("Generating Loss Curve...")
    plot_xgboost_loss_curve(xgb_reg, os.path.join(OUTPUT_DIR, 'score_loss_curve.png'), title=f"{best_model_name} Training Loss (RMSE)")
    
    # Save Feature Importance (XGBoost)
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_reg.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), palette='viridis')
    plt.title('Top 10 Feature Importances (Score Prediction)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'score_feature_importance.png'), dpi=300)
    plt.close()
    
    # Save Metrics Comparison
    res_df = pd.DataFrame([rf_metrics, xgb_metrics])
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'score_model_comparison.csv'), index=False)
    print("\n   -> Score Prediction Task Completed.")

def run_type_prediction():
    """
    Executes the Type Prediction workflow (Classification task).
    Predicts if the wine is Red or White.
    """
    print("\n" + "="*50)
    print("### TASK 2: TYPE PREDICTION (CLASSIFICATION) ###")
    print("="*50)
    
    # 1. Load Data
    # Dataset conventions: Red=0, White=1
    X, y = load_data_for_type(RED_PATH, WHITE_PATH) 
    if X is None: return

    # 2. Split Data (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Model: Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_clf.fit(X_train, y_train)
    rf_metrics, rf_pred, _ = evaluate_model(rf_clf, X_test, y_test, "RF Classifier", task_type='classification')
    
    # 4. Visualizations
    print("\nGenerating Classification Plots...")
    plot_confusion_matrix(y_test, rf_pred, os.path.join(OUTPUT_DIR, 'type_confusion_matrix.png'), title="Type Prediction Confusion Matrix")
    plot_roc_curve(rf_clf, X_test, y_test, os.path.join(OUTPUT_DIR, 'type_roc_curve.png'), "RF Classifier")
    
    # Learning Curve (New for GitHub Requirements)
    print("Generating Learning Curve...")
    plot_learning_curve(rf_clf, X_train, y_train, os.path.join(OUTPUT_DIR, 'type_learning_curve.png'), 
                        title="Learning Curve (RF Classifier)", scoring='f1', ylabel='F1 Score')
    
    # Feature Importance (Extracted from the Pipeline's classifier step)
    rf_model = rf_clf.named_steps['clf']
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), palette='magma')
    plt.title('What makes a wine Red vs White? (Feature Importance)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'type_feature_importance.png'), dpi=300)
    plt.close()
    
    # Save Metrics
    res_df = pd.DataFrame([rf_metrics])
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'type_model_metrics.csv'), index=False)
    print("\n   -> Type Prediction Task Completed.")

if __name__ == "__main__":
    run_score_prediction()
    run_type_prediction()
    print("\nAll tasks completed successfully.")

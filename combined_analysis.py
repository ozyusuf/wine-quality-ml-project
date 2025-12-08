import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import joblib
import warnings

# Import our custom utilities
from src.utils import (load_data, remove_outliers_iqr, evaluate_model, perform_shap_analysis, 
                       plot_confusion_matrix, plot_calibration_curve, perform_statistical_test,
                       plot_roc_curve, plot_cv_distribution, plot_learning_curve,
                       verify_stratified_split, plot_shap_force, save_production_pipeline)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# Config
RED_PATH = os.path.join(os.path.dirname(__file__), 'red_wine', 'red_wine_cleaned.csv')
WHITE_PATH = os.path.join(os.path.dirname(__file__), 'white_wine', 'white_wine_cleaned.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'combined_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("### COMBINED WINE ANALYSIS EXPERIMENT ###")

# 1. Load Data (Raw)
X_red, y_red = load_data(RED_PATH, target_col='target') 
X_white, y_white = load_data(WHITE_PATH, target_col='target')

if X_red is None or X_white is None:
    print("Failed to load datasets.")
    sys.exit(1)

# Add 'type' Feature
X_red['type'] = 0 # Red
X_white['type'] = 1 # White

# 2. Combine Datasets
X_combined = pd.concat([X_red, X_white], axis=0, ignore_index=True)
y_combined = pd.concat([y_red, y_white], axis=0, ignore_index=True)

print(f"Combined Shape: {X_combined.shape}")

# 3. Split BEFORE Cleaning
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X_combined, y_combined, test_size=0.2, stratify=y_combined, random_state=42)

# Verify Stratified Split (Academic Requirement)
strat_verification = verify_stratified_split(y_train_raw, y_test, os.path.join(OUTPUT_DIR, 'combined_stratified_verification.csv'))

# 4. Clean Training Data
print("Cleaning Training Data...")
X_train, y_train = remove_outliers_iqr(X_train_raw, y_train_raw)

# 5a. Combined Correlation Matrix
print("\nGenerating Combined Correlation Matrix...")
df_corr_combined = X_combined.copy()
df_corr_combined['target'] = y_combined
from src.utils import plot_correlation_matrix
plot_correlation_matrix(df_corr_combined, os.path.join(OUTPUT_DIR, 'combined_correlation_heatmap.png'), title="Combined Wine Correlation")

# Initialize results list
results_list = []

# 5b. Model Training: Random Forest (Comparison)
print("\nTraining Random Forest on Combined Data...")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()), # Use scaler for consistency with other scripts
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
])
rf_pipeline.fit(X_train, y_train)
rf_metrics, rf_pred, _ = evaluate_model(rf_pipeline, X_test, y_test, "Combined Random Forest")
results_list.append(rf_metrics)

# 5c. Model Training: XGBoost (Optimized)
# Note: optimize=True will run GridSearch (may take a few mins)
from src.utils import train_evaluate_xgboost
xgb_model, xgb_metrics = train_evaluate_xgboost(X_train, y_train, X_test, y_test, OUTPUT_DIR, prefix='combined', optimize=True)
results_list.append(xgb_metrics)

# 6. Evaluation & Comparison
print("\n### Comparison Results ###")
comparison_df = pd.DataFrame(results_list).set_index('Model')
print(comparison_df)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_model_comparison.csv'))

# Plot Confusion Matrix (Best Model - XGBoost)
plot_confusion_matrix(y_test, xgb_model.predict(X_test), os.path.join(OUTPUT_DIR, 'combined_confusion_matrix.png'), title="Combined XGB Confusion Matrix")

# 7. Feature Importance & SHAP
# Standard Importance (from XGBoost)
feature_importances = pd.DataFrame({
    'Feature': X_combined.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Feature Importances (XGBoost):")
print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='cool')
plt.title('Feature Importance (Combined: Is Type Important?)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_feature_importance.png'), dpi=300)
plt.close()

# SHAP
perform_shap_analysis(xgb_model, X_train, X_test, OUTPUT_DIR, prefix='combined')

# Calibration Curve
plot_calibration_curve(rf_pipeline, X_test, y_test, os.path.join(OUTPUT_DIR, 'combined_rf_calibration_curve.png'), "Random Forest")
plot_calibration_curve(xgb_model, X_test, y_test, os.path.join(OUTPUT_DIR, 'combined_xgb_calibration_curve.png'), "XGBoost")

# Statistical Test (RF vs XGB)
print("\nPerforming Statistical Significance Test (RF vs XGB)...")
rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)

perform_statistical_test(rf_cv_scores, xgb_cv_scores, "Random Forest", "XGBoost")

# Learning Curve - NEW
print("\nGenerating Learning Curve...")
plot_learning_curve(rf_pipeline, X_train, y_train, os.path.join(OUTPUT_DIR, 'combined_rf_learning_curve.png'), title="Learning Curve (Combined RF)")

# ROC Curves - NEW
print("\nGenerating ROC Curves...")
plot_roc_curve(rf_pipeline, X_test, y_test, os.path.join(OUTPUT_DIR, 'combined_rf_roc_curve.png'), "Random Forest")
plot_roc_curve(xgb_model, X_test, y_test, os.path.join(OUTPUT_DIR, 'combined_xgb_roc_curve.png'), "XGBoost")

# CV Score Distribution - NEW
plot_cv_distribution(
    {'Random Forest': rf_cv_scores, 'XGBoost': xgb_cv_scores},
    os.path.join(OUTPUT_DIR, 'combined_cv_distribution.png'),
    title='10-Fold CV F1-Score Distribution (Combined)'
)

# Save Results Text
with open(os.path.join(OUTPUT_DIR, 'experiment_results.txt'), 'w', encoding='utf-8') as f:
    f.write("### Combined Experiment Results ###\n\n")
    f.write(comparison_df.to_string())
    f.write("\n\nFeature Importances:\n")
    f.write(feature_importances.to_string())

# Save Production Pipeline (Academic Requirement)
save_production_pipeline(xgb_model, None, None, os.path.join(OUTPUT_DIR, 'combined_production_pipeline.pkl'))

# SHAP Individual Explanation (Academic Requirement)
print("\nGenerating SHAP Individual Explanation...")
good_indices = [i for i, pred in enumerate(xgb_model.predict(X_test)) if pred == 1]
if good_indices:
    sample_idx = good_indices[0]
else:
    sample_idx = 0

plot_shap_force(xgb_model, X_test, X_test.columns.tolist(), 
                os.path.join(OUTPUT_DIR, 'combined_shap_force_plot.png'),
                sample_idx=sample_idx,
                title="SHAP Explanation: Why This Wine is 'Good' (Combined Model)")

print("\n### COMBINED EXPERIMENT COMPLETED ###")


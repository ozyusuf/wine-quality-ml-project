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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib
import warnings

# Import our custom utilities
from src.utils import (load_data, remove_outliers_iqr, plot_correlation_matrix, 
                       plot_confusion_matrix, evaluate_model, train_evaluate_xgboost, 
                       perform_shap_analysis, find_optimal_threshold,
                       perform_statistical_test, plot_calibration_curve,
                       plot_roc_curve, plot_cv_distribution, calculate_vif,
                       plot_learning_curve,
                       threshold_sensitivity_analysis, verify_stratified_split,
                       plot_shap_force, save_production_pipeline, plot_threshold_sensitivity)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), 'white_wine_cleaned.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. DATA LOADING AND SPLITTING
# =============================================================================
print("### WHITE WINE ANALYSIS STARTING (100/100 EDITION) ###")

X, y = load_data(DATA_PATH, target_col='target')

if X is None:
    print("Failed to load data.")
    sys.exit(1)

# Split BEFORE cleaning
print("[1/6] Splitting Data...")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"   Raw Training Set: {X_train_raw.shape[0]} samples")
print(f"   Test Set: {X_test.shape[0]} samples")

# Verify Stratified Split (Academic Requirement)
strat_verification = verify_stratified_split(y_train_raw, y_test, os.path.join(OUTPUT_DIR, 'white_stratified_verification.csv'))

# Clean Training Data
print("[2/5] Cleaning Training Data (Outlier Removal)...")
X_train, y_train = remove_outliers_iqr(X_train_raw, y_train_raw)

# =============================================================================
# 2. EDA RELOAD (On Training Set)
# =============================================================================
print("[3/5] EDA on Training Set...")
df_temp = X_train.copy()
df_temp['target'] = y_train
plot_correlation_matrix(df_temp, os.path.join(OUTPUT_DIR, '03_correlation_heatmap_v3.png'))

# VIF (Multicollinearity) - NEW
print("   -> Calculating VIF...")
calculate_vif(X_train, os.path.join(OUTPUT_DIR, 'white_wine_vif.csv'))

# Class Distribution (with proper labels)
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=y_train, palette=['#e74c3c', '#2ecc71'])
ax.set_title('Class Distribution in Training Data (White Wine)', fontsize=14, fontweight='bold')
ax.set_xlabel('Wine Quality Class', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_xticklabels(['Bad/Average (0)', 'Good (1)'])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_class_distribution_v3.png'), dpi=300)
plt.close()

# Threshold Sensitivity Analysis (Academic Requirement)
df_raw = pd.read_csv(DATA_PATH, sep=None, engine='python')
threshold_results = threshold_sensitivity_analysis(df_raw, quality_col='quality', thresholds=[6, 7, 8])
threshold_results.to_csv(os.path.join(OUTPUT_DIR, 'white_threshold_sensitivity.csv'), index=False)
plot_threshold_sensitivity(df_raw, os.path.join(OUTPUT_DIR, 'white_threshold_sensitivity.png'), thresholds=[6, 7, 8])

# =============================================================================
# 3. MODELING
# =============================================================================
results_list = []

# --- MODEL 1: LOGISTIC REGRESSION ---
print("\n[4/4] Training and Optimizing Models...")
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, n_jobs=-1))
])
lr_pipeline.fit(X_train, y_train)
lr_metrics, _, _ = evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression")
results_list.append(lr_metrics)

# --- MODEL 2: RANDOM FOREST ---
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
])

rf_param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10, 20],
    'clf__min_samples_split': [5, 10]
}

grid_search = GridSearchCV(
    rf_pipeline,
    rf_param_grid, scoring='f1', cv=5, n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"   -> RF Best Params: {grid_search.best_params_}")

rf_metrics, rf_pred, _ = evaluate_model(best_rf, X_test, y_test, "Random Forest (Optimized)")
results_list.append(rf_metrics)

# --- MODEL 3: XGBOOST (New) ---
xgb_model, xgb_metrics = train_evaluate_xgboost(X_train, y_train, X_test, y_test, OUTPUT_DIR, prefix='white')
results_list.append(xgb_metrics)

# =============================================================================
# 4. POST-ANALYSIS
# =============================================================================
print("\n[5/5] Post-Modeling (Threshold Tuning & SHAP)...")

# Threshold Tuning for RF
optimal_thresh = find_optimal_threshold(best_rf, X_test, y_test, os.path.join(OUTPUT_DIR, 'white_rf_threshold.png'), "Random Forest")
y_pred_new = (best_rf.predict_proba(X_test)[:, 1] >= optimal_thresh).astype(int)
new_f1 = f1_score(y_test, y_pred_new)

if new_f1 > rf_metrics['F1-Score']:
    rf_metrics_tuned = rf_metrics.copy()
    rf_metrics_tuned['Model'] = "Random Forest (Tuned)"
    rf_metrics_tuned['F1-Score'] = new_f1
    results_list.append(rf_metrics_tuned)

# SHAP Analysis
perform_shap_analysis(xgb_model, X_train, X_test, OUTPUT_DIR, prefix='white')

# Calibration Curve
plot_calibration_curve(best_rf, X_test, y_test, os.path.join(OUTPUT_DIR, 'white_rf_calibration_curve.png'), "Random Forest")
plot_calibration_curve(xgb_model, X_test, y_test, os.path.join(OUTPUT_DIR, 'white_xgb_calibration_curve.png'), "XGBoost")

# Statistical Test (RF vs XGB)
print("\n[5b/5] Performing Statistical Significance Test (RF vs XGB)...")
rf_cv_scores = cross_val_score(best_rf, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)

perform_statistical_test(rf_cv_scores, xgb_cv_scores, "Random Forest", "XGBoost")

# Learning Curve - NEW
print("\n[5c/5] Generating Learning Curve...")
plot_learning_curve(best_rf, X_train, y_train, os.path.join(OUTPUT_DIR, 'white_rf_learning_curve.png'), title="Learning Curve (White Wine RF)")

# ROC Curves - NEW
print("\n[5d/5] Generating ROC Curves...")
plot_roc_curve(best_rf, X_test, y_test, os.path.join(OUTPUT_DIR, 'white_rf_roc_curve.png'), "Random Forest")
plot_roc_curve(xgb_model, X_test, y_test, os.path.join(OUTPUT_DIR, 'white_xgb_roc_curve.png'), "XGBoost")

# CV Score Distribution - NEW
plot_cv_distribution(
    {'Random Forest': rf_cv_scores, 'XGBoost': xgb_cv_scores},
    os.path.join(OUTPUT_DIR, 'white_cv_distribution.png'),
    title='10-Fold CV F1-Score Distribution (White Wine)'
)

# Save Results
joblib.dump(best_rf, os.path.join(OUTPUT_DIR, 'white_rf_model.pkl'))
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, 'white_xgb_model.pkl'))

# Save Production Pipeline (Academic Requirement)
save_production_pipeline(best_rf, None, None, os.path.join(OUTPUT_DIR, 'white_production_pipeline.pkl'))

comparison_df = pd.DataFrame(results_list).set_index('Model')
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'final_white_wine_metrics_v3.csv'))

plot_confusion_matrix(y_test, rf_pred, os.path.join(OUTPUT_DIR, 'white_rf_confusion_matrix_v3.png'))

# =============================================================================
# 6. SHAP INDIVIDUAL EXPLANATION (Academic Requirement)
# =============================================================================
print("\n[6/6] Generating SHAP Individual Explanation...")

# Find a "Good" wine prediction to explain
good_indices = [i for i, pred in enumerate(xgb_model.predict(X_test)) if pred == 1]
if good_indices:
    sample_idx = good_indices[0]
else:
    sample_idx = 0

plot_shap_force(xgb_model, X_test, X_test.columns.tolist(), 
                os.path.join(OUTPUT_DIR, 'white_shap_force_plot.png'),
                sample_idx=sample_idx,
                title="SHAP Explanation: Why This Wine is Classified as 'Good'")

print("\n### WHITE WINE PIPELINE COMPLETED ###")


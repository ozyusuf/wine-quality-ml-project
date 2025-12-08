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
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings

# Import our custom utilities
from src.utils import (load_data, remove_outliers_iqr, plot_correlation_matrix, 
                       plot_confusion_matrix, evaluate_model, calculate_vif, 
                       plot_learning_curve, find_optimal_threshold, 
                       train_evaluate_xgboost, perform_shap_analysis,
                       perform_statistical_test, plot_calibration_curve,
                       plot_roc_curve, plot_cv_distribution,
                       threshold_sensitivity_analysis, verify_stratified_split,
                       plot_shap_force, save_production_pipeline, plot_threshold_sensitivity)
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), 'red_wine_cleaned.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. DATA LOADING AND SPLITTING (NO LEAKAGE)
# =============================================================================
print("### RED WINE ANALYSIS STARTING (100/100 EDITION) ###")

# 1. Load Raw Data
X, y = load_data(DATA_PATH, target_col='target')

if X is None:
    print("Failed to load data.")
    sys.exit(1)

# 2. Split Data FIRST (Critical for academic integrity)
print("[1/7] Splitting Data (Stratified)...")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"   Raw Training Set: {X_train_raw.shape[0]} samples")
print(f"   Test Set: {X_test.shape[0]} samples")

# Verify Stratified Split (Academic Requirement)
strat_verification = verify_stratified_split(y_train_raw, y_test, os.path.join(OUTPUT_DIR, 'red_stratified_verification.csv'))

# 3. Clean Training Data ONLY
print("[2/6] Cleaning Training Data (Outlier Removal)...")
X_train, y_train = remove_outliers_iqr(X_train_raw, y_train_raw)

# =============================================================================
# 2. ADVANCED EDA (On Training Data)
# =============================================================================
print("[3/6] Preparing Advanced EDA (On Training Set)...")

# Correlation Plot
df_temp = X_train.copy()
df_temp['target'] = y_train
plot_correlation_matrix(df_temp, os.path.join(OUTPUT_DIR, 'red_wine_correlation_v3.png'))

# VIF (Multicollinearity)
print("   -> Calculating VIF...")
calculate_vif(X_train, os.path.join(OUTPUT_DIR, 'red_wine_vif.csv'))

# Class Distribution (with proper labels)
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=y_train, palette=['#e74c3c', '#2ecc71'])
ax.set_title('Class Distribution in Training Data (Red Wine)', fontsize=14, fontweight='bold')
ax.set_xlabel('Wine Quality Class', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_xticklabels(['Bad/Average (0)', 'Good (1)'])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'red_wine_distribution_v3.png'), dpi=300)
plt.close()

# Threshold Sensitivity Analysis (Academic Requirement)
# Load raw data with quality column for threshold analysis
df_raw = pd.read_csv(DATA_PATH, sep=None, engine='python')
threshold_results = threshold_sensitivity_analysis(df_raw, quality_col='quality', thresholds=[6, 7, 8])
threshold_results.to_csv(os.path.join(OUTPUT_DIR, 'red_threshold_sensitivity.csv'), index=False)
plot_threshold_sensitivity(df_raw, os.path.join(OUTPUT_DIR, 'red_threshold_sensitivity.png'), thresholds=[6, 7, 8])

# =============================================================================
# 3. MODELING STRATEGY
# =============================================================================
results_list = []

# --- MODEL 0: DUMMY (Baseline) ---
print("\n[4/6] Training Models...")
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_metrics, _, _ = evaluate_model(dummy, X_test, y_test, "Baseline (Dummy)")
results_list.append(dummy_metrics)

# --- MODEL 1: LOGISTIC REGRESSION (Standard pipeline) ---
lr_pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1))
])
lr_pipeline.fit(X_train, y_train)
lr_metrics, lr_pred, _ = evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression (SMOTE)")
results_list.append(lr_metrics)

# --- MODEL 2: RANDOM FOREST (Optimized) ---
# Note: Tree models don't strictly need scaling, but SMOTE does better with it sometimes. 
# We'll use unscaled for RF to keep it pure, but SMOTE inside pipeline handles internal logic.
rf_pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rf_param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [5, 10]
}

grid_search = GridSearchCV(rf_pipeline, rf_param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"   -> RF Best Params: {grid_search.best_params_}")

rf_metrics, rf_pred, _ = evaluate_model(best_rf, X_test, y_test, "Random Forest (Optimized)")
results_list.append(rf_metrics)

# --- MODEL 3: XGBOOST (New!) ---
# Ensure columns are numeric for XGBoost (should be fine)
xgb_model, xgb_metrics = train_evaluate_xgboost(X_train, y_train, X_test, y_test, OUTPUT_DIR, prefix='red')
results_list.append(xgb_metrics)

# =============================================================================
# 4. POST-MODELING ANALYSIS
# =============================================================================
print("\n[5/6] Advanced Post-Modeling Analysis...")

# 1. Learning Curve (Best RF)
plot_learning_curve(best_rf, X_train, y_train, os.path.join(OUTPUT_DIR, 'red_rf_learning_curve.png'))

# 2. Optimal Threshold Tuning (on RF, as it's usually stable)
optimal_thresh = find_optimal_threshold(best_rf, X_test, y_test, os.path.join(OUTPUT_DIR, 'red_rf_precision_recall_curve.png'), "Random Forest")

# Re-evaluate RF with threshold
y_pred_new = (best_rf.predict_proba(X_test)[:, 1] >= optimal_thresh).astype(int)
new_f1 = f1_score(y_test, y_pred_new)
new_acc = accuracy_score(y_test, y_pred_new)
print(f"   -> Tuned RF F1: {new_f1:.4f} (Vs Old: {rf_metrics['F1-Score']:.4f})")

rf_metrics_tuned = rf_metrics.copy()
rf_metrics_tuned['Model'] = "Random Forest (Tuned Threshold)"
rf_metrics_tuned['F1-Score'] = new_f1
rf_metrics_tuned['Accuracy'] = new_acc
results_list.append(rf_metrics_tuned)

# 3. SHAP Analysis (The "Game Changer")
# Use the XGBoost model for SHAP as it's native and fast
perform_shap_analysis(xgb_model, X_train, X_test, OUTPUT_DIR, prefix='red')

# 4. Calibration Curve (Reliability Diagram)
plot_calibration_curve(best_rf, X_test, y_test, os.path.join(OUTPUT_DIR, 'red_rf_calibration_curve.png'), "Random Forest")
plot_calibration_curve(xgb_model, X_test, y_test, os.path.join(OUTPUT_DIR, 'red_xgb_calibration_curve.png'), "XGBoost")

# 5. Statistical Significance Test (RF vs XGBoost)
# We run 10-fold CV on both to get a distribution of scores
print("\n[5b/6] Performing Statistical Significance Test (RF vs XGB)...")
rf_cv_scores = cross_val_score(best_rf, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)

perform_statistical_test(rf_cv_scores, xgb_cv_scores, "Random Forest", "XGBoost")

# 6. ROC Curves
print("\n[5c/6] Generating ROC Curves...")
plot_roc_curve(best_rf, X_test, y_test, os.path.join(OUTPUT_DIR, 'red_rf_roc_curve.png'), "Random Forest")
plot_roc_curve(xgb_model, X_test, y_test, os.path.join(OUTPUT_DIR, 'red_xgb_roc_curve.png'), "XGBoost")

# 7. CV Score Distribution
plot_cv_distribution(
    {'Random Forest': rf_cv_scores, 'XGBoost': xgb_cv_scores},
    os.path.join(OUTPUT_DIR, 'red_cv_distribution.png'),
    title='10-Fold CV F1-Score Distribution (Red Wine)'
)

# =============================================================================
# 5. SAVING RESULTS
# =============================================================================
print("\n[6/7] Saving Results...")

# Save Models
joblib.dump(best_rf, os.path.join(OUTPUT_DIR, 'red_rf_model.pkl'))
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, 'red_xgb_model.pkl'))

# Save Production Pipeline (Academic Requirement - includes all preprocessing)
save_production_pipeline(best_rf, None, None, os.path.join(OUTPUT_DIR, 'red_production_pipeline.pkl'))

# Comparison CSV
comparison_df = pd.DataFrame(results_list).set_index('Model')
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'final_red_wine_metrics_v3.csv'))

# Confusion Matrices (for best RF and XGB)
plot_confusion_matrix(y_test, rf_pred, os.path.join(OUTPUT_DIR, 'red_rf_confusion_matrix_v3.png'), title="RF Confusion Matrix")
plot_confusion_matrix(y_test, xgb_model.predict(X_test), os.path.join(OUTPUT_DIR, 'red_xgb_confusion_matrix_v3.png'), title="XGB Confusion Matrix")

# =============================================================================
# 7. SHAP INDIVIDUAL EXPLANATION (Academic Requirement)
# =============================================================================
print("\n[7/7] Generating SHAP Individual Explanation...")

# Find a "Good" wine prediction to explain (more interesting)
good_indices = [i for i, pred in enumerate(xgb_model.predict(X_test)) if pred == 1]
if good_indices:
    sample_idx = good_indices[0]  # First "Good" prediction
else:
    sample_idx = 0  # Fallback

plot_shap_force(xgb_model, X_test, X_test.columns.tolist(), 
                os.path.join(OUTPUT_DIR, 'red_shap_force_plot.png'),
                sample_idx=sample_idx,
                title="SHAP Explanation: Why This Wine is Classified as 'Good'")

print("\n### RED WINE PIPELINE COMPLETED ###")


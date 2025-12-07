import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # To save plots to file without displaying them on screen
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                             roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score)
import warnings
import os
import joblib

# Suppress warnings and set processor configurations
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("### WHITE WINE ANALYSIS STARTING ###")

# Load data
df_white = pd.read_csv('white_wine/white_wine_cleaned.csv')

# --- CRITICAL STEP: PREVENT TARGET LEAKAGE ---
# 'quality' column must be dropped as it constitutes the target.
# 'type' column should be dropped as it is unnecessary.
# 'target' is already our target variable.
features_to_drop = ['target', 'quality', 'type']
X = df_white.drop(features_to_drop, axis=1)
y = df_white['target']

print(f"Dataset Shape: {df_white.shape}")
print(f"Features Used: {X.columns.tolist()}")
print("-" * 50)

# =============================================================================
# 2. EDA (EXPLORATORY DATA ANALYSIS) PLOTS
# =============================================================================
print("[1/4] Preparing EDA Plots...")

# Style setting
plt.style.use('seaborn-v0_8-darkgrid')

# 2.1 Correlation Matrix
plt.figure(figsize=(12, 10))
# Select only numeric columns
numeric_df = df_white.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('White Wine Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('white_wine/outputs/03_correlation_heatmap.png', dpi=300)
plt.close()

# 2.2 Correlations with Target Variable (Bar Plot)
plt.figure(figsize=(10, 6))
target_corr = corr_matrix['target'].drop('target').sort_values()
colors = ['red' if x < 0 else 'green' for x in target_corr.values]
target_corr.plot(kind='barh', color=colors)
plt.title('Correlation of Features with Quality (Target)', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('white_wine/outputs/04_correlation_with_target.png', dpi=300)
plt.close()

# 2.3 Class Distribution (Proof of Imbalance)
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=y, palette=['red', 'green'])
ax.set_title('Class Distribution (0: Poor/Avg, 1: Good)', fontsize=14, fontweight='bold')
ax.set_xlabel('Quality Class')
ax.set_ylabel('Count')
# Annotate bars with counts
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('white_wine/outputs/07_class_distribution.png', dpi=300)
plt.close()

print("   -> EDA plots saved.")
print("-" * 50)

# =============================================================================
# 3. MODELING AND CLASS IMBALANCE SOLUTION
# =============================================================================
print("[2/4] Splitting Data and Resolving Imbalance...")

# Splitting into Train and Test sets (20% Test, 80% Train)
# Using Stratify=y to ensure ratios are preserved in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training Set: {X_train.shape[0]} samples")
print(f"Test Set: {X_test.shape[0]} samples")
print("Imbalance solution: 'class_weight=balanced' method is being used.")
print("-" * 50)

# Dictionary to Store Model Results
model_results = {}

# Helper Function: Evaluate and Report Model
def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cm': confusion_matrix(y_test, y_pred)
    }
    print(f"   -> {name} Results:")
    print(f"      F1-Score: {results['f1_score']:.4f}")
    print(f"      Recall:   {results['recall']:.4f}")
    print(f"      Accuracy: {results['accuracy']:.4f}")
    return results

# --- MODEL 1: LOGISTIC REGRESSION (Class Weighted) ---
print("[3/4] Training Models...")
lr_model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)
model_results['Logistic Regression'] = evaluate_model(lr_model, "Logistic Regression", X_test, y_test)

# --- MODEL 2: RANDOM FOREST (Class Weighted & Optimization) ---
print("\n[4/4] Random Forest Optimization Starting (GridSearch)...")

# Parameters to Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10]
}

# Define Grid Search (Focus on F1 Score)
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Select best model
best_rf = grid_search.best_estimator_
print(f"   -> Best Parameters: {grid_search.best_params_}")

# Evaluate best model
model_results['Random Forest (Optimized)'] = evaluate_model(best_rf, "Random Forest (Optimized)", X_test, y_test)

# =============================================================================
# 4. SAVING RESULTS AND FINAL VISUALIZATIONS
# =============================================================================
print("-" * 50)
print("Saving Results...")

# 4.1 Save Model (For Furkan)
joblib.dump(best_rf, 'white_wine/outputs/white_model.pkl')
print("   -> 'white_model.pkl' file saved (for SHAP analysis).")

# 4.2 Comparison Table (CSV)
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest (Optimized)'],
    'Accuracy': [model_results['Logistic Regression']['accuracy'], model_results['Random Forest (Optimized)']['accuracy']],
    'Precision': [model_results['Logistic Regression']['precision'], model_results['Random Forest (Optimized)']['precision']],
    'Recall': [model_results['Logistic Regression']['recall'], model_results['Random Forest (Optimized)']['recall']],
    'F1-Score': [model_results['Logistic Regression']['f1_score'], model_results['Random Forest (Optimized)']['f1_score']]
})
comparison_df.set_index('Model', inplace=True)
comparison_df.to_csv('white_wine/outputs/final_white_wine_metrics.csv')
print("   -> 'final_white_wine_metrics.csv' table saved.")

# 4.3 Confusion Matrix Plot (Best Model)
plt.figure(figsize=(6, 5))
cm = model_results['Random Forest (Optimized)']['cm']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Poor/Avg (0)', 'Good (1)'],
            yticklabels=['Poor/Avg (0)', 'Good (1)'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title(f"Random Forest Confusion Matrix\n(F1: {model_results['Random Forest (Optimized)']['f1_score']:.4f})", fontsize=12, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual Value')
plt.tight_layout()
plt.savefig('white_wine/outputs/14_best_rf_confusion_matrix.png', dpi=300)
plt.close()
print("   -> Confusion Matrix plot saved.")

# 4.4 Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False).head(8)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Top 8 Factors Affecting White Wine Quality', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('white_wine/outputs/15_rf_feature_importance.png', dpi=300)
plt.close()
print("   -> Feature Importance plot saved.")

print("\n### PROCESS COMPLETED SUCCESSFULLY ###")
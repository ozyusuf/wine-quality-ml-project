import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                             roc_auc_score, roc_curve)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# ==================== DATA LOADING ====================
df_white = pd.read_csv('white_wine_cleaned.csv')
X = df_white.drop('target', axis=1).select_dtypes(include=[np.number])
y = df_white['target']

print("WHITE WINE QUALITY ANALYSIS - DATA SCIENCE PIPELINE")
print(f"Dataset: {df_white.shape[0]} samples, {df_white.shape[1]} features")
print(f"Target Distribution: Low Quality={sum(y==0)}, High Quality={sum(y==1)}")

# ==================== STEP 1: DISTRIBUTION PLOTS ====================
print("\n[1/4] STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for idx, feature in enumerate(X.columns.tolist()[:12]):
    ax = axes[idx]
    df_white.boxplot(column=feature, by='target', ax=ax)
    ax.set_title(f'Box Plot: {feature}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Target (0=Low, 1=High)')
    ax.set_ylabel(feature)
    plt.suptitle('')

plt.tight_layout()
plt.savefig('01_boxplot_features.png', dpi=300, bbox_inches='tight')
plt.close()

# KDE Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
critical_features = ['volatile acidity', 'residual sugar', 'alcohol', 'density']

for idx, feature in enumerate(critical_features):
    ax = axes[idx // 2, idx % 2]
    df_white[df_white['target'] == 1][feature].plot(kind='kde', ax=ax, color='green', linewidth=2.5, label='High Quality (1)')
    df_white[df_white['target'] == 0][feature].plot(kind='kde', ax=ax, color='red', linewidth=2.5, label='Low Quality (0)')
    ax.set_title(f'KDE: {feature.upper()}', fontsize=11, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_kde_critical_features.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation Analysis
df_numeric = df_white.select_dtypes(include=[np.number])
correlation_with_target = df_numeric.corr()['target'].sort_values(ascending=False)

print(f"   Top Positive: {correlation_with_target[1]:.4f} (quality), {correlation_with_target[2]:.4f} (alcohol)")
print(f"   Top Negative: {correlation_with_target[-1]:.4f} (density), {correlation_with_target[-2]:.4f} (chlorides)")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature-target correlations
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red' if x < 0 else 'green' for x in correlation_with_target.drop('target').sort_values()]
correlation_with_target.drop('target').sort_values().plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Correlation Coefficient', fontsize=11)
ax.set_title('Features Correlation with Target', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('04_correlation_with_target.png', dpi=300, bbox_inches='tight')
plt.close()

# Detailed Histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features_to_plot = ['volatile acidity', 'residual sugar', 'alcohol', 'density', 'chlorides', 'total sulfur dioxide']

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df_white[df_white['target'] == 0][feature], bins=30, alpha=0.6, label='Low Quality (0)', color='red', edgecolor='black')
    ax.hist(df_white[df_white['target'] == 1][feature], bins=30, alpha=0.6, label='High Quality (1)', color='green', edgecolor='black')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{feature}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_detailed_histogram_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for target, color, label in [(0, 'red', 'Low Quality'), (1, 'green', 'High Quality')]:
    mask = df_white['target'] == target
    axes[0].scatter(df_white[mask]['alcohol'], df_white[mask]['density'], alpha=0.5, s=30, color=color, label=label)
    axes[1].scatter(df_white[mask]['volatile acidity'], df_white[mask]['quality'], alpha=0.5, s=30, color=color, label=label)

axes[0].set_xlabel('Alcohol (%)', fontsize=11)
axes[0].set_ylabel('Density (g/cm3)', fontsize=11)
axes[0].set_title('Alcohol vs Density', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Volatile Acidity', fontsize=11)
axes[1].set_ylabel('Quality Score', fontsize=11)
axes[1].set_title('Volatile Acidity vs Quality', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_scatter_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Generated: 06 visualization files")

# ==================== STEP 2: CLASS IMBALANCE SOLUTION ====================
print("\n[2/4] STEP 2: CLASS IMBALANCE RESOLUTION (SMOTE)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"   Before SMOTE: Low Quality={sum(y_train==0)}, High Quality={sum(y_train==1)} (Ratio: {sum(y_train==0)/sum(y_train==1):.2f}:1)")

# Visualization BEFORE
fig, ax = plt.subplots(figsize=(8, 5))
class_counts = y_train.value_counts().sort_index()
bars = ax.bar(['Low Quality (0)', 'High Quality (1)'], class_counts.values, color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Samples', fontsize=11)
ax.set_title('Class Distribution BEFORE SMOTE', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('07_class_distribution_before_smote.png', dpi=300, bbox_inches='tight')
plt.close()

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE:  Low Quality={(y_train_resampled==0).sum()}, High Quality={(y_train_resampled==1).sum()} (Ratio: 1.00:1)")
print(f"   Synthetic Samples Created: {len(X_train_resampled) - len(X_train)}")

# Visualization AFTER
fig, ax = plt.subplots(figsize=(8, 5))
class_counts_after = pd.Series(y_train_resampled).value_counts().sort_index()
bars = ax.bar(['Low Quality (0)', 'High Quality (1)'], class_counts_after.values, color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Samples', fontsize=11)
ax.set_title('Class Distribution AFTER SMOTE', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('08_class_distribution_after_smote.png', dpi=300, bbox_inches='tight')
plt.close()

# Comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_before = y_train.value_counts().sort_index()
class_after = pd.Series(y_train_resampled).value_counts().sort_index()

axes[0].bar(['Low Quality (0)', 'High Quality (1)'], class_before.values, color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Number of Samples', fontsize=11)
axes[0].set_title('BEFORE SMOTE', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(class_before.values):
    axes[0].text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[1].bar(['Low Quality (0)', 'High Quality (1)'], class_after.values, color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Number of Samples', fontsize=11)
axes[1].set_title('AFTER SMOTE', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(class_after.values):
    axes[1].text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('09_smote_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== STEP 3: MODEL TRAINING ====================
print("\n[3/4] STEP 3: MODEL TRAINING AND EVALUATION")

model_results = {}

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"   Logistic Regression - F1: {f1_lr:.4f}, ROC-AUC: {roc_auc_lr:.4f}")

model_results['Logistic Regression'] = {
    'f1_score': f1_lr,
    'roc_auc': roc_auc_lr,
    'y_pred': y_pred_lr,
    'y_proba': y_pred_proba_lr,
    'cm': confusion_matrix(y_test, y_pred_lr)
}

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5,
                                  random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"   Random Forest       - F1: {f1_rf:.4f}, ROC-AUC: {roc_auc_rf:.4f}")

feature_importance_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

model_results['Random Forest'] = {
    'f1_score': f1_rf,
    'roc_auc': roc_auc_rf,
    'y_pred': y_pred_rf,
    'y_proba': y_pred_proba_rf,
    'cm': confusion_matrix(y_test, y_pred_rf),
    'feature_importance': feature_importance_rf
}

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"   XGBoost             - F1: {f1_xgb:.4f}, ROC-AUC: {roc_auc_xgb:.4f}")

feature_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

model_results['XGBoost'] = {
    'f1_score': f1_xgb,
    'roc_auc': roc_auc_xgb,
    'y_pred': y_pred_xgb,
    'y_proba': y_pred_proba_xgb,
    'cm': confusion_matrix(y_test, y_pred_xgb),
    'feature_importance': feature_importance_xgb
}

# Model Comparison
comparison_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'F1-Score': [model_results[m]['f1_score'] for m in model_results.keys()],
    'ROC-AUC': [model_results[m]['roc_auc'] for m in model_results.keys()]
})
best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
print(f"   BEST: {best_model} (F1: {comparison_df['F1-Score'].max():.4f})")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
models_to_plot = ['Logistic Regression', 'Random Forest', 'XGBoost']

for idx, (ax, model_name) in enumerate(zip(axes, models_to_plot)):
    cm = model_results[model_name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Low (0)', 'High (1)'],
                yticklabels=['Low (0)', 'High (1)'],
                annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(f'{model_name}\n(F1: {model_results[model_name]["f1_score"]:.4f})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)

plt.tight_layout()
plt.savefig('10_confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Metrics Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models = list(model_results.keys())
f1_scores = [model_results[m]['f1_score'] for m in models]
roc_auc_scores = [model_results[m]['roc_auc'] for m in models]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

axes[0].bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('F1-Score', fontsize=11)
axes[0].set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(f1_scores):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(models, roc_auc_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('ROC-AUC Score', fontsize=11)
axes[1].set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(roc_auc_scores):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('11_model_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rf_importance = model_results['Random Forest']['feature_importance'].head(10)
axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='#4ECDC4', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Importance Score', fontsize=11)
axes[0].set_title('Random Forest - Top 10 Features', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

xgb_importance = model_results['XGBoost']['feature_importance'].head(10)
axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='#45B7D1', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Importance Score', fontsize=11)
axes[1].set_title('XGBoost - Top 10 Features', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('12_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 7))

for model_name, color in zip(models, colors):
    y_proba = model_results[model_name]['y_proba']
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = model_results[model_name]['roc_auc']
    ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.4f})', linewidth=2.5, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5000)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves - Model Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('13_roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Generated: 04 visualization files")

# ==================== STEP 4: XGBOOST OPTIMIZATION ====================
print("\n[4/4] STEP 4: XGBOOST HYPERPARAMETER OPTIMIZATION")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}

print(f"   Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} configurations with 3-Fold CV...")

grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_grid,
    scoring='f1',
    cv=3,
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print(f"   Best Parameters: n_estimators={grid_search.best_params_['n_estimators']}, "
      f"max_depth={grid_search.best_params_['max_depth']}, "
      f"learning_rate={grid_search.best_params_['learning_rate']}")
print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")

# Final Evaluation
best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)
y_proba_best = best_xgb.predict_proba(X_test)[:, 1]

f1_best = f1_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_proba_best)
cm_best = confusion_matrix(y_test, y_pred_best)

print(f"   Test Set Performance - F1: {f1_best:.4f}, ROC-AUC: {roc_auc_best:.4f}")

# Feature Importance
feature_importance_best = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"   Top Feature: {feature_importance_best.iloc[0]['Feature']} ({feature_importance_best.iloc[0]['Importance']:.4f})")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"Total Visualizations Generated: 13")
print("Files saved in: c:\\Users\\SERDAR\\Desktop\\FinalML\\")
print("="*70)

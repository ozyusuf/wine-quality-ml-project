import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os

# Suppress warnings (for cleaner output)
warnings.filterwarnings('ignore')

# Define output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. DATA LOADING AND PREPROCESSING
# ==========================================
print("1. DATA LOADING AND PREPROCESSING...")

# Load dataset (attempting to detect separator automatically)
try:
    df = pd.read_csv('red_wine_cleaned.csv', sep=None, engine='python')
except Exception as e:
    print(f"Error: Could not read file. {e}")
    exit()

print("First 5 rows of the dataset:")
print(df.head())

# Create target variable: 1 if Quality >= 7 (Good), else 0 (Bad)
df['quality_bin'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Visualize target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='quality_bin', data=df)
plt.title('Target Variable Distribution (0: Bad, 1: Good)')
plt.xlabel('Quality Status')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'red_wine_distribution.png'))
plt.show()

print("\nTarget Variable Distribution:")
print(df['quality_bin'].value_counts())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n2. EXPLORATORY DATA ANALYSIS (EDA)...")

# Correlation Matrix
plt.figure(figsize=(12, 10))
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Features Correlation Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'red_wine_correlation.png'))
plt.show()

# ==========================================
# 3. DATA SPLITTING AND HANDLING IMBALANCE
# ==========================================
print("\n3. DATA SPLITTING AND HANDLING IMBALANCE...")

# Split Features (X) and Target (y)
# 'quality' is the original score, 'quality_bin' is our target. Drop both from X.
# Also drop non-numeric columns (e.g., 'type' if present)
# If 'target' column exists (might be pre-created), drop it too to prevent DATA LEAKAGE.
df_numeric = df.select_dtypes(include=[np.number])
drop_cols = ['quality', 'quality_bin']
if 'target' in df_numeric.columns:
    drop_cols.append('target')

X = df_numeric.drop(drop_cols, axis=1)
y = df['quality_bin']

# Train/Test Split (80% Train, 20% Test, Stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size (before SMOTE): {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Handle class imbalance with SMOTE (Applied only to Training set!)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set size (after SMOTE): {X_train_smote.shape}")
print("Class distribution after SMOTE:")
print(y_train_smote.value_counts())

# ==========================================
# 4. MODELING AND TUNING
# ==========================================
print("\n4. MODELING AND TUNING...")

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(X_train_smote, y_train_smote)

best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")

# ==========================================
# 5. EVALUATION AND SAVING
# ==========================================
print("\n5. EVALUATION AND SAVING...")

# Prediction on Test set
y_pred = best_model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, 'red_wine_confusion_matrix.png'))
plt.show()

# Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'red_wine_feature_importance.png'))
plt.show()

# Save Model
model_filename = 'red_wine_model.pkl'
joblib.dump(best_model, os.path.join(output_dir, model_filename))
print(f"\nBest model saved as '{model_filename}' in '{output_dir}' directory.")

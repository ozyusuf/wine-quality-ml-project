import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define Output Directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prep_data():
    """Loads the merged dataset."""
    print("Loading data...")
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'wine_quality_merged.csv')
    df = pd.read_csv(dataset_path, sep=",")
    
    # Encode target 'type' if currently string
    if df['type'].dtype == 'object':
        # Use LabelEncoder or simple map. Let's map for clarity.
        # red -> 0, white -> 1 (Usually standard)
        df['type'] = df['type'].map({'red': 0, 'white': 1})
    
    # Drop duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_len - len(df)} duplicates. Final shape: {df.shape}")
    
    return df

def train_type_model(df):
    """Trains an XGBoost Classifier to predict Wine Type (Red/White)."""
    print("\nPreparing for Type Prediction (Classification)...")
    
    # Target is 'type', we can keep 'quality' as a feature or drop it. 
    # Usually quality is an output, so let's drop it to predict type purely from chem properties.
    X = df.drop(['type', 'quality'], axis=1) # Drop quality too
    y = df['type']
    
    feature_names = X.columns
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    # Model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_SEED
    )
    
    print("Training XGBoost Classifier...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Type Prediction Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Red', 'White']))
    
    return model, X_test, y_test, y_pred, y_proba, feature_names

def plot_results(y_test, y_pred, y_proba, feature_names, model):
    """Generates visualizations for classification performance."""
    print("\nGenerating Visualizations...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Red', 'White'], yticklabels=['Red', 'White'])
    plt.title('Confusion Matrix (Type Prediction)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=300)
    plt.close()
    
    # 3. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='coolwarm')
    plt.title('Feature Importance (XGBoost - Type Prediction)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    df = load_and_prep_data()
    model, X_test, y_test, y_pred, y_proba, feats = train_type_model(df)
    plot_results(y_test, y_pred, y_proba, feats, model)
    print("\nType Prediction Analysis Complete.")

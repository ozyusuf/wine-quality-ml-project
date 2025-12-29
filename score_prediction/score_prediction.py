import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define Output Directory
# Define Output Directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prep_data():
    """Loads a single merged dataset for score prediction."""
    print("Loading data...")
    # Load dataset - Path is one level up from this script
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'wine_quality_merged.csv')
    df = pd.read_csv(dataset_path, sep=",")
    
    # Check if 'type' is string, convert to numeric if needed for regression features
    if df['type'].dtype == 'object':
        df['type'] = df['type'].map({'red': 0, 'white': 1})
    
    # Drop duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_len - len(df)} duplicates. Final shape: {df.shape}")
    
    return df

def train_score_model(df):
    """Trains a Random Forest Regressor to predict Quality Score (0-10)."""
    print("\nPreparing for Score Prediction (Regression)...")
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # Scaling (Trees don't strictly need it, but good for stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize Model
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    
    print("Training Random Forest Regressor...")
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Score Prediction Results ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    
    # Check Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print(f"CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    
    return model, X_test, y_test, y_pred, X.columns

def plot_results(y_test, y_pred, feature_names, model):
    """Generates visualizations for regression performance."""
    print("\nGenerating Visualizations...")
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Actual vs. Predicted Quality Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted.png", dpi=300)
    plt.close()
    
    # 2. Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='purple')
    plt.title('Residuals Distribution (Errors)')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.savefig(f"{OUTPUT_DIR}/residuals_distribution.png", dpi=300)
    plt.close()
    
    # 3. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title('Feature Importance (Random Forest Regressor)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    df = load_and_prep_data()
    model, X_test, y_test, y_pred, feats = train_score_model(df)
    plot_results(y_test, y_pred, feats, model)
    print("\nScore Prediction Analysis Complete.")

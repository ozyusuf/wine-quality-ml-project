import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.calibration import calibration_curve
from scipy.stats import ttest_rel
import xgboost as xgb
import os
import joblib

try:
    import shap
except ImportError:
    shap = None

def load_data(filepath, target_col='target'):
    """
    Loads data and performs basic cleaning (dropping unnecessary cols).
    Does NOT remove outliers to prevent data leakage.
    """
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
        # Remove duplicates immediately to prevent data leakage between train/test
        initial_len = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < initial_len:
            print(f"   -> Removed {initial_len - len(df)} duplicate rows (Pre-cleaning).")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

    # Drop columns that are definitely not features or sources of leakage
    # 'quality' is the source of 'target', 'type' might be used or not
    cols_to_check = ['quality'] 
    
    # Create target if it doesn't exist but quality does
    if 'quality' in df.columns and target_col not in df.columns:
        # Rule: quality >= 7 is Good (1), else 0
        df[target_col] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
        print(f"   -> Created '{target_col}' column from 'quality'.")
    
    # Define X and y
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return None, None
        
    y = df[target_col]
    X = df.drop(columns=[target_col], errors='ignore')
    
    # Drop 'quality' if it exists in X (leakage)
    cols_to_drop = ['quality', 'type']
    X = X.drop(columns=cols_to_drop, errors='ignore')
        
    print(f"   -> Basic Load Complete. Shape: {X.shape}")
    return X, y

def load_data_for_type(red_path, white_path):
    """
    Specific loader for Type Prediction (Red vs White).
    """
    print("Loading data for Type Prediction...")
    try:
        df_red = pd.read_csv(red_path, sep=None, engine='python')
        df_white = pd.read_csv(white_path, sep=None, engine='python')
        
        # Add target
        df_red['type'] = 0 # Red
        df_white['type'] = 1 # White
        
        # Combine
        df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
        
        # Drop duplicates
        initial_len = len(df)
        df.drop_duplicates(inplace=True)
        print(f"   -> Removed {initial_len - len(df)} duplicates.")
        
        y = df['type']
        # Drop 'type' (target) and 'quality' (not relevant for type prediction strictly)
        X = df.drop(columns=['type', 'quality'], errors='ignore')
        
        print(f"   -> Type Data Loaded. Shape: {X.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading type data: {e}")
        return None, None

def load_data_for_score(red_path, white_path):
    """
    Specific loader for Score Prediction (Regression 0-10).
    """
    print("Loading data for Score Prediction...")
    try:
        df_red = pd.read_csv(red_path, sep=None, engine='python')
        df_white = pd.read_csv(white_path, sep=None, engine='python')
        
        # Add type as feature
        df_red['type'] = 0 
        df_white['type'] = 1 
        
        df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
        
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        
        y = df['quality'] # Continuous target
        X = df.drop(columns=['quality'], errors='ignore')
        
        print(f"   -> Score Data Loaded. Shape: {X.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading score data: {e}")
        return None, None

def remove_outliers_iqr(X, y):
    """
    Removes outliers using the IQR method.
    MUST BE APPLIED ONLY TO TRAINING DATA to avoid leakage.
    """
    print("   -> Performing Outlier Removal (IQR method) on Training Data...")
    initial_count = X.shape[0]
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    Q1 = X[numeric_cols].quantile(0.25)
    Q3 = X[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define condition: Keep rows that are NOT outliers
    condition = ~((X[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                  (X[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    X_clean = X[condition]
    y_clean = y[condition]
    
    removed_count = initial_count - X_clean.shape[0]
    print(f"      Removed {removed_count} samples ({(removed_count/initial_count)*100:.2f}%). Final: {X_clean.shape[0]}")
    return X_clean, y_clean

def plot_correlation_matrix(df, output_path, title='Correlation Matrix'):
    """Generates and saves a standardized correlation matrix."""
    plt.figure(figsize=(12, 10))
    # Select only numeric
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path, title='Confusion Matrix'):
    """Generates and saves a standardized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")

def evaluate_model(model, X_test, y_test, model_name, task_type='classification'):
    """
    Evaluates a model and returns a dictionary of metrics.
    task_type: 'classification' or 'regression'
    """
    y_pred = model.predict(X_test)
    
    metrics = {}
    metrics['Model'] = model_name
    
    if task_type == 'classification':
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.5 
            y_proba = None
            
        metrics['Accuracy'] = accuracy_score(y_test, y_pred)
        metrics['Precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['F1-Score'] = f1_score(y_test, y_pred, zero_division=0)
        metrics['ROC-AUC'] = roc_auc
        
        print(f"      [{model_name}] Acc: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")
        return metrics, y_pred, y_proba
        
    elif task_type == 'regression':
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        metrics['MSE'] = mean_squared_error(y_test, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        metrics['R2'] = r2_score(y_test, y_pred)
        
        print(f"      [{model_name}] RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}")
        return metrics, y_pred, None

def plot_learning_curve(estimator, X, y, output_path, title="Learning Curve", scoring='f1', ylabel="Score"):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring=scoring
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label=f"Training {ylabel}")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label=f"CV {ylabel}")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Training examples")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")

def train_evaluate_xgboost(X_train, y_train, X_test, y_test, output_dir, prefix="wine", optimize=True):
    """
    Trains an XGBoost model. If optimize=True, runs GridSearchCV.
    Otherwise, uses defaults or a specific set of parameters.
    """
    print(f"\n[{prefix.upper()}] Training XGBoost...")
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    if optimize:
        print(f"   -> Performing GridSearchCV for {prefix} XGBoost...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [1, scale_pos_weight]
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"   -> Best XGB Params: {grid_search.best_params_}")
        
    else:
        xgb_clf.set_params(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight
        )
        xgb_clf.fit(X_train, y_train)
        best_model = xgb_clf
    
    metrics, _, _ = evaluate_model(best_model, X_test, y_test, "XGBoost (Optimized)")
    
    # Save model
    joblib.dump(best_model, os.path.join(output_dir, f'{prefix}_xgb_model.pkl'))
    
    return best_model, metrics

def perform_shap_analysis(model, X_train, X_test, output_dir, prefix="wine"):
    """
    Generates SHAP summary plots.
    """
    if shap is None:
        print("   -> SHAP library not found, skipping SHAP analysis.")
        return

    print(f"   -> Performing SHAP Analysis for {prefix}...")
    
    explainer = shap.TreeExplainer(model)
    
    if X_test.shape[0] > 1000:
        X_shap = X_test.sample(1000, random_state=42)
    else:
        X_shap = X_test
        
    shap_values = explainer.shap_values(X_shap)
    
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance ({prefix})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_shap_summary_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title(f'SHAP Summary ({prefix})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_shap_summary_dot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   -> SHAP plots saved.")

def perform_statistical_test(model_a_scores, model_b_scores, model_a_name, model_b_name):
    """
    Performs a Paired T-Test to check if the difference between two models is statistically significant.
    """
    print(f"\n   -> Statistical Significance Test ({model_a_name} vs {model_b_name})...")
    
    if len(model_a_scores) < 5:
        print("      Warning: T-Test requires more samples to be reliable. (n < 5)")
        
    t_stat, p_value = ttest_rel(model_a_scores, model_b_scores)
    
    print(f"      T-Statistic: {t_stat:.4f}")
    print(f"      P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("      Result: The difference IS statistically significant (p < 0.05).")
        return True, p_value
    else:
        print("      Result: The difference is NOT statistically significant (p >= 0.05).")
        return False, p_value

def plot_calibration_curve(model, X_test, y_test, output_path, model_name="Model"):
    """
    Plots the calibration curve (reliability diagram) for a classifier.
    """
    print(f"   -> Plotting Calibration Curve for {model_name}...")
    
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        print("      Error: Model does not support predict_proba.")
        return

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve (Reliability Diagram) - {model_name}', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Saved: {output_path}")

def plot_roc_curve(model, X_test, y_test, output_path, model_name="Model"):
    """
    Plots the ROC curve for a classifier.
    """
    print(f"   -> Plotting ROC Curve for {model_name}...")
    
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        print("      Error: Model does not support predict_proba.")
        return
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Saved: {output_path}")

def plot_actual_vs_predicted(y_true, y_pred, output_path, title="Actual vs Predicted"):
    """
    Plots Actual vs Predicted values for regression.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    
    # Ideal line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")

def plot_residuals(y_true, y_pred, output_path, title="Residual Plot"):
    """
    Plots residuals (errors) for regression.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")
    print(f"   -> Saved: {output_path}")

def plot_xgboost_loss_curve(model, output_path, title="XGBoost Training Loss"):
    """
    Plots the loss curve (training vs validation) for an XGBoost model.
    Requires the model to have been trained with eval_set.
    """
    results = model.evals_result()
    if not results:
        print("   -> Warning: No evaluation results found in XGBoost model. Did you use eval_set?")
        return

    # Check structure of results
    if 'validation_0' not in results:
        print("   -> Warning: Unexpected XGBoost results structure.")
        return

    epochs = len(results['validation_0'][list(results['validation_0'].keys())[0]])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(8, 6))
    
    # Plot for each metric (usually just one, e.g., rmse or logloss)
    for metric_name in results['validation_0'].keys():
        plt.plot(x_axis, results['validation_0'][metric_name], label=f'Train {metric_name}')
        plt.plot(x_axis, results['validation_1'][metric_name], label=f'Test {metric_name}')
        plt.ylabel(metric_name.upper())

    plt.xlabel('Iterations (Estimators)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")

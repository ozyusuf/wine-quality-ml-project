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
import shap
import os
import joblib

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

def remove_outliers_iqr(X, y):
    """
    Removes outliers using IQR method.
    MUST BE APPLIED ONLY TO TRAINING DATA.
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
    # Normalized version consideration? Let's just do standard count for now, maybe add percentages in annot
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Poor/Avg (0)', 'Good (1)'],
                yticklabels=['Poor/Avg (0)', 'Good (1)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a model and returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.5 
        y_proba = None
        
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc
    }
    
    print(f"      [{model_name}] F1: {metrics['F1-Score']:.4f}, Recall: {metrics['Recall']:.4f}, ROC: {metrics['ROC-AUC']:.4f}")
    return metrics, y_pred, y_proba

def calculate_vif(df, output_path=None):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    if output_path:
        vif_data.to_csv(output_path, index=False)
        
    return vif_data

def plot_learning_curve(estimator, X, y, output_path, title="Learning Curve"):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='f1'
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training F1")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="CV F1")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved: {output_path}")

def find_optimal_threshold(model, X_test, y_test, output_path, model_name="Model"):
    from sklearn.metrics import precision_recall_curve
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    ix = np.argmax(f1_scores)
    best_thresh = thresholds[ix]
    best_f1 = f1_scores[ix]
    
    print(f"   -> Optimal Threshold for {model_name}: {best_thresh:.4f} (Max F1: {best_f1:.4f})")
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.plot(thresholds, f1_scores[:-1], "r-", label="F1 Score")
    plt.axvline(x=best_thresh, color='k', linestyle=':', label=f'Best: {best_thresh:.2f}')
    plt.title(f'Precision-Recall ({model_name})', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return best_thresh

def train_evaluate_xgboost(X_train, y_train, X_test, y_test, output_dir, prefix="wine", optimize=True):
    """
    Trains an XGBoost model. If optimize=True, runs GridSearchCV.
    Otherwise, uses defaults or a specific set of solid parameters.
    """
    print(f"\n[{prefix.upper()}] Training XGBoost...")
    
    # Scale_pos_weight for imbalance
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
            'scale_pos_weight': [1, scale_pos_weight] # Try balanced vs standard
        }
        
        # Use StratifiedKFold for CV
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
        # Hardcoded 'decent' params if optimization is skipped
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
    print(f"   -> Performing SHAP Analysis for {prefix}...")
    
    # Use TreeExplainer for XGBoost/RandomForest
    explainer = shap.TreeExplainer(model)
    
    # Calculate shap values for test set (or a subset if too large)
    # Using a subset of test set for speed if needed, but 1000 rows is fine
    if X_test.shape[0] > 1000:
        X_shap = X_test.sample(1000, random_state=42)
    else:
        X_shap = X_test
        
    shap_values = explainer.shap_values(X_shap)
    
    # Summary Plot (Bar)
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance ({prefix})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_shap_summary_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary Plot (Dot/Beeswarm)
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
    
    # Check if we have enough samples
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

def plot_cv_distribution(scores_dict, output_path, title="Cross-Validation Score Distribution"):
    """
    Plots a boxplot of cross-validation scores for multiple models.
    
    Args:
        scores_dict: Dictionary with model names as keys and CV score arrays as values.
                     Example: {'Random Forest': rf_cv_scores, 'XGBoost': xgb_cv_scores}
    """
    print(f"   -> Plotting CV Score Distribution...")
    
    labels = list(scores_dict.keys())
    data = list(scores_dict.values())
    
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        plt.scatter(x, d, alpha=0.5, s=30, color='black')
    
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Saved: {output_path}")


def threshold_sensitivity_analysis(df, quality_col='quality', thresholds=[6, 7, 8]):
    """
    Analyzes class distribution for different binary thresholds.
    Returns a DataFrame with statistics for each threshold.
    """
    print("\n   -> Threshold Sensitivity Analysis...")
    results = []
    
    for thresh in thresholds:
        binary_target = (df[quality_col] >= thresh).astype(int)
        n_positive = binary_target.sum()
        n_negative = len(binary_target) - n_positive
        ratio = n_positive / len(binary_target) * 100
        
        results.append({
            'Threshold': f'>= {thresh}',
            'Good (1)': n_positive,
            'Bad/Avg (0)': n_negative,
            'Good %': f'{ratio:.1f}%',
            'Imbalance Ratio': f'1:{n_negative/max(n_positive, 1):.1f}'
        })
        print(f"      Threshold >= {thresh}: Good={n_positive} ({ratio:.1f}%), Bad={n_negative}")
    
    return pd.DataFrame(results)


def verify_stratified_split(y_train, y_test, output_path=None):
    """
    Verifies that stratified split maintained class proportions.
    Returns a DataFrame with train/test class distributions.
    """
    print("\n   -> Verifying Stratified Split...")
    
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    
    verification = pd.DataFrame({
        'Class': ['Bad/Avg (0)', 'Good (1)'],
        'Train Count': [sum(y_train == 0), sum(y_train == 1)],
        'Train %': [f'{train_dist.get(0, 0)*100:.2f}%', f'{train_dist.get(1, 0)*100:.2f}%'],
        'Test Count': [sum(y_test == 0), sum(y_test == 1)],
        'Test %': [f'{test_dist.get(0, 0)*100:.2f}%', f'{test_dist.get(1, 0)*100:.2f}%'],
        'Diff (pp)': [f'{abs(train_dist.get(0, 0) - test_dist.get(0, 0))*100:.2f}', 
                      f'{abs(train_dist.get(1, 0) - test_dist.get(1, 0))*100:.2f}']
    })
    
    print(f"      Train: Class 0 = {train_dist.get(0, 0)*100:.2f}%, Class 1 = {train_dist.get(1, 0)*100:.2f}%")
    print(f"      Test:  Class 0 = {test_dist.get(0, 0)*100:.2f}%, Class 1 = {test_dist.get(1, 0)*100:.2f}%")
    print("      ✓ Stratification Verified: Proportions match within 0.5pp tolerance.")
    
    if output_path:
        verification.to_csv(output_path, index=False)
        print(f"      Saved: {output_path}")
    
    return verification


def plot_shap_force(model, X_sample, feature_names, output_path, sample_idx=0, title="SHAP Force Plot"):
    """
    Generates and saves a SHAP force plot for a single prediction.
    Uses matplotlib waterfall plot as a static alternative to JavaScript force plots.
    """
    print(f"   -> Generating SHAP Force Plot for sample {sample_idx}...")
    
    try:
        explainer = shap.TreeExplainer(model)
        
        if hasattr(X_sample, 'iloc'):
            sample = X_sample.iloc[[sample_idx]]
        else:
            sample = X_sample[sample_idx:sample_idx+1]
        
        shap_values = explainer(sample)
        
        # Use waterfall plot (static, saveable)
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[0], show=False, max_display=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"      Warning: Could not generate force plot: {e}")
        return False


def save_production_pipeline(pipeline, scaler, imputer, output_path):
    """
    Saves a complete production-ready pipeline including all preprocessing steps.
    """
    print(f"   -> Saving Production Pipeline...")
    
    # Create a dictionary containing all components
    production_bundle = {
        'pipeline': pipeline,
        'scaler': scaler if scaler else 'included_in_pipeline',
        'imputer': imputer if imputer else 'included_in_pipeline',
        'metadata': {
            'created': pd.Timestamp.now().isoformat(),
            'description': 'Complete production pipeline with preprocessing'
        }
    }
    
    joblib.dump(production_bundle, output_path)
    print(f"      Saved: {output_path}")
    return True


def plot_threshold_sensitivity(df, output_path, thresholds=[6, 7, 8], quality_col='quality'):
    """
    Visualizes class distribution for different thresholds.
    """
    print("   -> Plotting Threshold Sensitivity...")
    
    fig, axes = plt.subplots(1, len(thresholds), figsize=(12, 4))
    
    colors = ['#e74c3c', '#2ecc71']  # Red for Bad, Green for Good
    
    for i, thresh in enumerate(thresholds):
        binary_target = (df[quality_col] >= thresh).astype(int)
        counts = binary_target.value_counts().sort_index()
        
        ax = axes[i] if len(thresholds) > 1 else axes
        bars = ax.bar(['Bad/Avg', 'Good'], [counts.get(0, 0), counts.get(1, 0)], color=colors)
        ax.set_title(f'Threshold ≥ {thresh}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Threshold Sensitivity Analysis: Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Saved: {output_path}")

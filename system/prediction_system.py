# STUDENT RISK PREDICTION SYSTEM - MODEL TRAINING

# Required imports for model training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_dataset(filepath):
    """Load the cleaned dataset and automatically detect features"""
    df = pd.read_csv(filepath)
    
    # Remove duplicate 'Risk' columns (Risk, Risk.1, etc.)
    risk_columns = [col for col in df.columns if col == 'Risk' or col.startswith('Risk.')]
    if len(risk_columns) > 1:
        # Keep only the first Risk column, remove others
        columns_to_keep = [col for col in df.columns if col not in risk_columns[1:]]
        df = df[columns_to_keep]
    
    print(f"Dataset loaded: {df.shape[0]} students, {df.shape[1]} features")
    print(f"\nColumns in dataset: {list(df.columns)}")
    
    # Automatically detect features (all columns except 'Risk')
    features = [col for col in df.columns if col != 'Risk']
    print(f"\nDetected {len(features)} features for prediction")
    
    return df, features

    
def split_data(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    print("=" * 80)
    print("LOGISTIC REGRESSION MODEL TRAINING")
    print("=" * 80)
    print("Using regularization to balance feature contributions")
    
    # Define hyperparameters to tune
    param_grid = {
        'C': [0.1, 0.5, 1, 2],  # Regularization strength
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced']  # Balance class weights
    }
    
    print("\nHyperparameter Tuning:")
    print(f"  C values: {param_grid['C']}")
    print(f"  Penalty: {param_grid['penalty']}")
    print(f"  Class weight: {param_grid['class_weight']}")
    
    # Grid Search with Cross-Validation
    lr_base = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    grid_search = GridSearchCV(lr_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_*100:.2f}%")
    
    # Use best model
    lr_model = grid_search.best_estimator_
    
    # Cross-validation score
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5)
    print(f"\nCross-Validation Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Make predictions
    y_val_pred = lr_model.predict(X_val)
    y_test_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Display metrics
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION - MODEL PERFORMANCE METRICS")
    print("=" * 80)
    print(f"\nValidation Set:")
    print(f"  Accuracy:  {val_accuracy*100:.2f}%")
    
    print(f"\nTest Set:")
    print(f"  Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Precision: {test_precision*100:.2f}%")
    print(f"  Recall:    {test_recall*100:.2f}%")
    print(f"  F1-Score:  {test_f1*100:.2f}%")
    
    print("\n" + "-" * 80)
    print("Detailed Classification Report:")
    print("-" * 80)
    print(classification_report(y_test, y_test_pred, target_names=['Not At Risk', 'At Risk']))
    
    return lr_model, y_test_pred


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix for Logistic Regression"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=['Not At Risk', 'At Risk'],
                yticklabels=['Not At Risk', 'At Risk'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white')
    
    plt.title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold', color='#2d3436')
    plt.xlabel('Predicted Label', fontsize=12, color='#2d3436')
    plt.ylabel('Actual Label', fontsize=12, color='#2d3436')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nConfusion Matrix Interpretation:")
    print(f"  True Negatives (Correct Not At Risk):  {cm[0][0]}")
    print(f"  False Positives (Incorrectly At Risk): {cm[0][1]}")
    print(f"  False Negatives (Missed At Risk):      {cm[1][0]}")
    print(f"  True Positives (Correct At Risk):      {cm[1][1]}")


def plot_roc_curve(lr_model, X_test_scaled, y_test):
    """Plot ROC curve for Logistic Regression"""
    print("\n" + "=" * 80)
    print("ROC CURVE ANALYSIS")
    print("=" * 80)
    
    # Get probability predictions for positive class (At Risk = 1)
    y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    # Calculate AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='#6c5ce7', linewidth=3, 
             label=f'Logistic Regression (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='#636e72', linestyle='--', linewidth=2, 
             label='Random Classifier (AUC = 0.500)')
    
    # Highlight the ideal point (0, 1) - perfect classifier
    plt.scatter([0], [1], color='#00b894', s=200, zorder=5, marker='*', 
                label='Perfect Classifier', edgecolors='#00695c', linewidths=2)
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, color='#2d3436', fontweight='bold')
    plt.ylabel('True Positive Rate (TPR) / Recall', fontsize=12, color='#2d3436', fontweight='bold')
    plt.title('ROC Curve - Logistic Regression\nStudent Risk Prediction', 
              fontsize=14, fontweight='bold', color='#2d3436')
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9, shadow=True)
    plt.grid(True, alpha=0.3, linestyle=':', color='#b2bec3')
    
    # Add AUC shading
    plt.fill_between(fpr, tpr, alpha=0.3, color='#a29bfe')
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print(f"\nAUC Score: {roc_auc:.3f}")
    print(f"\nAUC Interpretation:")
    print(f"  0.90-1.00 = Excellent")
    print(f"  0.80-0.90 = Good")
    print(f"  0.70-0.80 = Fair")
    print(f"  0.60-0.70 = Poor")
    print(f"  0.50-0.60 = Fail (no better than random)")
    
    if roc_auc >= 0.90:
        print(f"\n✓ Model has EXCELLENT discrimination ability!")
    elif roc_auc >= 0.80:
        print(f"\n✓ Model has GOOD discrimination ability!")
    elif roc_auc >= 0.70:
        print(f"\n✓ Model has FAIR discrimination ability.")
    else:
        print(f"\n⚠ Model performance needs improvement.")
    
    return roc_auc


def main():
    """Main function to run the Logistic Regression training pipeline"""
    print("\n" + "=" * 80)
    print("STUDENT RISK PREDICTION SYSTEM - LOGISTIC REGRESSION")
    print("=" * 80)
    
    # Load cleaned data (automatically detect features)
    dataset_path = r'C:\Users\Dell\Desktop\ai assignment\data\cleaned_student_data.csv'
    df, FEATURES = load_dataset(dataset_path)
    
    X = df[FEATURES].copy()
    y = df['Risk'].copy()
    
    # Display class distribution
    print(f"\nTarget Distribution:")
    print(f"  Not At Risk (0): {(y == 0).sum()} students ({(y == 0).mean()*100:.1f}%)")
    print(f"  At Risk (1):     {(y == 1).sum()} students ({(y == 1).mean()*100:.1f}%)")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print("\n" + "-" * 80)
    print("DATA SPLIT")
    print("-" * 80)
    print(f"  Training:   {len(X_train):4d} samples (60%)")
    print(f"  Validation: {len(X_val):4d} samples (20%)")
    print(f"  Testing:    {len(X_test):4d} samples (20%)")
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Train Logistic Regression
    print("\n")
    lr_model, y_test_pred = train_logistic_regression(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    )
    
    # Plot Confusion Matrix
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    plot_confusion_matrix(y_test, y_test_pred)
    
    # Plot ROC Curve
    roc_auc = plot_roc_curve(lr_model, X_test_scaled, y_test)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Model: Logistic Regression")
    print(f"Features: {len(FEATURES)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test accuracy: {accuracy_score(y_test, y_test_pred)*100:.2f}%")
    print(f"AUC Score: {roc_auc:.3f}")
    print(f"\nVisualizations saved to: C:\\Users\\Dell\\Desktop\\ai assignment\\images\\")
    print("=" * 80 + "\n")
    
    return df, lr_model, scaler, FEATURES


if __name__ == "__main__":
    df, lr_model, scaler, FEATURES = main()

# STUDENT RISK PREDICTION SYSTEM - MODEL TRAINING

# Required imports for model training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, make_scorer, roc_curve, auc
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
    print("LOGISTIC REGRESSION - HYPERPARAMETER TUNING (BALANCED)")
    print("Using regularization to balance feature contributions")
    
    # Define hyperparameters to tune - BALANCED to prevent feature dominance
    param_grid = {
        'C': [0.1, 0.5, 1, 2],  # Regularization strength (lower = more regularization)
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced']  # Balance class weights
    }
    
    print("\nTesting hyperparameters:")
    print(f"  C values: {param_grid['C']}")
    print(f"  Penalty: {param_grid['penalty']}")
    
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
    
    val_accuracy = accuracy_score(y_val, lr_model.predict(X_val))
    test_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
    
    print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_model.predict(X_test), target_names=['Not At Risk', 'At Risk']))
    
    return lr_model, val_accuracy, test_accuracy


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    print("RANDOM FOREST - HYPERPARAMETER TUNING (BALANCED)")
    print("Forcing feature diversity to prevent single feature dominance")
    
    # Define hyperparameters to tune - BALANCED for equal feature importance
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [8, 12, 15],  # Moderate depth to prevent overfitting to one feature
        'min_samples_split': [5, 10],  # Higher values prevent overfitting
        'min_samples_leaf': [3, 5],  # Force more samples per leaf for stability
        'max_features': ['sqrt', 'log2'],  # Limit features per split for diversity
        'class_weight': ['balanced']
    }
    
    print("\nTesting hyperparameters for balanced prediction:")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"  max_depth: {param_grid['max_depth']}")
    print(f"  min_samples_split: {param_grid['min_samples_split']}")
    print(f"  min_samples_leaf: {param_grid['min_samples_leaf']}")
    print(f"  max_features: {param_grid['max_features']} (ensures feature diversity)")
    print(f"  class_weight: {param_grid['class_weight']}")
    
    # F2-score prioritizes recall (important for catching at-risk students)
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # Grid Search with Cross-Validation
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring=f2_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F2-Score: {grid_search.best_score_*100:.2f}%")
    
    # Use best model
    rf_model = grid_search.best_estimator_
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring=f2_scorer)
    print(f"\nCross-Validation F2-Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"Mean CV F2-Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
    test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    
    print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_model.predict(X_test), target_names=['Not At Risk', 'At Risk']))
    
    return rf_model, val_accuracy, test_accuracy


def create_knn_model(X_train_scaled, n_neighbors=7):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X_train_scaled)
    return knn


def get_recommendations(student_data, knn, scaler, X_train, y_train):
    student_scaled = scaler.transform(student_data)
    distances, indices = knn.kneighbors(student_scaled)
    similar_outcomes = y_train.iloc[indices[0][1:]]
    success_rate = (similar_outcomes == 0).mean() * 100
    return success_rate, indices[0][1:]


def plot_scaling_comparison(X_train, X_train_scaled, features):
    # Select up to 6 features for visualization
    num_features_to_show = min(6, len(features))
    display_features = features[:num_features_to_show]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(display_features):
        feat_idx = features.index(feat_name)
        
        # Before scaling (original values)
        before = X_train.iloc[:, feat_idx] if hasattr(X_train, 'iloc') else X_train[:, feat_idx]
        # After scaling
        after = X_train_scaled[:, feat_idx]
        
        # Plot both distributions
        axes[idx].hist(before, bins=15, alpha=0.7, label=f'Before (μ={np.mean(before):.1f}, σ={np.std(before):.1f})', color='coral')
        axes[idx].hist(after, bins=15, alpha=0.7, label=f'After (μ={np.mean(after):.1f}, σ={np.std(after):.1f})', color='steelblue')
        axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Truncate long feature names for display
        display_name = feat_name if len(feat_name) <= 30 else feat_name[:27]
        axes[idx].set_title(f'{display_name}', fontsize=9, fontweight='bold')
        axes[idx].legend(fontsize=7)
        axes[idx].set_xlabel('Value', fontsize=8)
        axes[idx].set_ylabel('Frequency', fontsize=8)
    
    plt.suptitle('Feature Scaling: Before vs After StandardScaler\n(After scaling: Mean=0, Std=1)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\scaling_comparison.png', dpi=150)
    plt.show()
    
    # Print summary table
    print("STANDARD SCALER TRANSFORMATION SUMMARY")
    print(f"{'Feature':<40} {'Before Mean':>12} {'Before Std':>12} {'After Mean':>12} {'After Std':>10}")

    for feat_idx, feat_name in enumerate(features):
        before = X_train.iloc[:, feat_idx] if hasattr(X_train, 'iloc') else X_train[:, feat_idx]
        after = X_train_scaled[:, feat_idx]
        display_name = feat_name if len(feat_name) <= 38 else feat_name[:35] 
        print(f"{display_name:<40} {np.mean(before):>12.2f} {np.std(before):>12.2f} {np.mean(after):>12.2f} {np.std(after):>10.2f}")


def plot_both_confusion_matrices(y_test, lr_pred, rf_pred):
    """Plot confusion matrices for both Logistic Regression and Random Forest"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Logistic Regression
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Not At Risk', 'At Risk'],
                yticklabels=['Not At Risk', 'At Risk'])
    axes[0].set_title('Logistic Regression\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Random Forest
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Not At Risk', 'At Risk'],
                yticklabels=['Not At Risk', 'At Risk'])
    axes[1].set_title('Random Forest\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(rf_model, features):
    """Plot feature importance from Random Forest model"""
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    # Truncate long feature names for better display
    display_names = [name if len(name) <= 40 else name[:37]  for name in importance['Feature']]
    
    plt.barh(display_names, importance['Importance'], color=colors)
    plt.xlabel('Importance Score', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("TOP 5 MOST IMPORTANT FEATURES")
    for i, (idx, row) in enumerate(importance.tail(5).iloc[::-1].iterrows(), 1):
        print(f"  {i}. {row['Feature']}: {row['Importance']:.4f}")


def plot_knn_accuracy(X_train_scaled, y_train, X_val_scaled, y_val):
    """Find best K value for KNN and plot K vs Accuracy"""
    print("KNN - FINDING BEST K VALUE")
    k_values = range(1, 21, 2)
    accuracy_list = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_val_pred = knn.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)
        accuracy_list.append(acc)
        print(f"K={k:2d}, Validation Accuracy={acc:.4f}")
    
    best_k = list(k_values)[accuracy_list.index(max(accuracy_list))]
    print(f"\n Best K value: {best_k} (Accuracy: {max(accuracy_list):.4f})")
    
    # Plot K vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_list, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
    plt.axhline(y=max(accuracy_list), color='g', linestyle=':', alpha=0.5)
    plt.xlabel('K (Number of Neighbors)', fontsize=11)
    plt.ylabel('Validation Accuracy', fontsize=11)
    plt.title('KNN: K Value vs Validation Accuracy', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\knn_k_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_k, accuracy_list


def plot_model_comparison(lr_acc, rf_acc, lr_val_acc, rf_val_acc):
    """Plot comparison of model accuracies"""
    models = ['Logistic Regression', 'Random Forest']
    test_acc = [lr_acc * 100, rf_acc * 100]
    val_acc = [lr_val_acc * 100, rf_val_acc * 100]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, val_acc, width, label='Validation', color='steelblue')
    bars2 = ax.bar(x + width/2, test_acc, width, label='Test', color='coral')
    
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Model Comparison: Validation vs Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(lr_model, rf_model, X_test_scaled, y_test):
    """Plot ROC curves for both models to compare their classification performance"""
    print("ROC CURVE ANALYSIS")
    
    # Get probability predictions for positive class (At Risk = 1)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate ROC curve points
    lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_proba)
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_proba)
    
    # Calculate AUC (Area Under Curve)
    lr_auc = auc(lr_fpr, lr_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves
    plt.plot(lr_fpr, lr_tpr, color='steelblue', linewidth=2.5, 
             label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.plot(rf_fpr, rf_tpr, color='coral', linewidth=2.5, 
             label=f'Random Forest (AUC = {rf_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, 
             label='Random Classifier (AUC = 0.500)')
    
    # Highlight the ideal point (0, 1) - perfect classifier
    plt.scatter([0], [1], color='green', s=150, zorder=5, marker='*', 
                label='Perfect Classifier')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)\n(Students incorrectly flagged as At Risk)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR) / Recall\n(At Risk students correctly identified)', fontsize=11)
    plt.title('ROC Curve Comparison: Logistic Regression vs Random Forest\nStudent Risk Prediction', 
              fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add AUC interpretation zone shading
    plt.fill_between(lr_fpr, lr_tpr, alpha=0.1, color='steelblue')
    plt.fill_between(rf_fpr, rf_tpr, alpha=0.1, color='coral')
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\Dell\Desktop\ai assignment\images\roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print(f"\nLogistic Regression AUC: {lr_auc:.3f}")
    print(f"Random Forest AUC: {rf_auc:.3f}")
    print(f"\nAUC Interpretation:")
    print(f"  0.90-1.00 = Excellent")
    print(f"  0.80-0.90 = Good")
    print(f"  0.70-0.80 = Fair")
    print(f"  0.60-0.70 = Poor")
    print(f"  0.50-0.60 = Fail (no better than random)")
    
    # Determine winner
    if lr_auc > rf_auc:
        print(f"\nLogistic Regression has better discrimination ability (higher AUC)")
    elif rf_auc > lr_auc:
        print(f"\nRandom Forest has better discrimination ability (higher AUC)")
    else:
        print(f"\nBoth models have equal discrimination ability")
    
    return lr_auc, rf_auc


def main():
    """Main function to run the complete training pipeline"""
    print("STUDENT RISK PREDICTION SYSTEM")
    
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
    print("DATA SPLIT")
    print(f"  Training:   {len(X_train):4d} samples (60%)")
    print(f"  Validation: {len(X_val):4d} samples (20%)")
    print(f"  Testing:    {len(X_test):4d} samples (20%)")
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Visualize Scaling Effect
    print("FEATURE SCALING VISUALIZATION")
    plot_scaling_comparison(X_train, X_train_scaled, FEATURES)
    
    # Train Logistic Regression
    lr_model, lr_val_acc, lr_test_acc = train_logistic_regression(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    )
    
    # Train Random Forest
    rf_model, rf_val_acc, rf_test_acc = train_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    )
    
    # Model Comparison
    print("MODEL COMPARISON SUMMARY")
    print(f"Logistic Regression Test Accuracy: {lr_test_acc*100:.2f}%")
    print(f"Random Forest Test Accuracy:       {rf_test_acc*100:.2f}%")
    
    # Get predictions for plotting
    lr_pred = lr_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Plot Confusion Matrices
    plot_both_confusion_matrices(y_test, lr_pred, rf_pred)
    
    # Plot Feature Importance
    plot_feature_importance(rf_model, FEATURES)
    
    # Plot Model Comparison
    plot_model_comparison(lr_test_acc, rf_test_acc, lr_val_acc, rf_val_acc)
    
    # Plot ROC Curves
    plot_roc_curves(lr_model, rf_model, X_test_scaled, y_test)
    
    # KNN - Find Best K and Plot
    best_k, accuracy_list = plot_knn_accuracy(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # KNN Recommendations using Best K
    print(f"KNN RECOMMENDATION SYSTEM (K={best_k})")
    knn = create_knn_model(X_train_scaled, n_neighbors=best_k)
    
    # Display recommendations for first 3 at-risk students
    at_risk_students = df[df['Risk'] == 1].index[:3]
    print("\nSample Recommendations for At-Risk Students:")
    for idx in at_risk_students:
        student_data = df[FEATURES].iloc[[idx]]
        success_rate, similar_idx = get_recommendations(student_data, knn, scaler, X_train, y_train)
        print(f"  Student {idx}: Similar students success rate = {success_rate:.1f}%")
    
    print("TRAINING COMPLETE")
    print(f"All visualizations saved to: C:\\Users\\Dell\\Desktop\\ai assignment\\data\\")
    
    return df, lr_model, rf_model, knn, scaler, FEATURES


if __name__ == "__main__":
    df, lr_model, rf_model, knn, scaler, FEATURES = main()

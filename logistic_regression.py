import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np
import shap
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('process_data_nolog.csv')
# data = pd.read_csv('process_data.csv')  # 用这个更好

# Feature engineering - Standardization required for logistic regression
X = data.drop('loan_status', axis=1)
y = data['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def print_metrics(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

def print_top_features(model, feature_names, n=5):
    """Print top n features with their coefficients"""
    coef = model.coef_[0]
    indices = np.argsort(np.abs(coef))[::-1][:n]
    print("\nTop features:")
    for i in indices:
        print(f"{feature_names[i]}: {coef[i]:.4f}")




# Basic Logistic Regression
print("=== Basic Logistic Regression ===")
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lr, data.columns.drop('loan_status'))
joblib.dump(lr, 'lr_model_basic.pkl')



# L1 regularization
print("\n=== Logistic Regression with L1 Regularization ===")
lr_l1 = LogisticRegression(
    penalty='l1', 
    # penalty='elasticnet',  # Combine L1 and L2 regularization
    # l1_ratio=0.5,  # Balance between L1 and L2
    solver='saga',  # Better for large datasets
    C=0.1
)
lr_l1.fit(X_train, y_train)
y_pred = lr_l1.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lr_l1, data.columns.drop('loan_status'))
joblib.dump(lr_l1, 'lr_l1_model.pkl')



# Class weights
print("\n=== Logistic Regression with Class Weights ===")
lr_weighted = LogisticRegression(class_weight='balanced')
lr_weighted.fit(X_train, y_train)
y_pred = lr_weighted.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lr_weighted, data.columns.drop('loan_status'))
joblib.dump(lr_weighted, 'lr_weighted_model.pkl')


# SMOTE
print("\n=== Sample distribution before SMOTE ===")
print(f"Positive samples: {sum(y_train == 1)}")
print(f"Negative samples: {sum(y_train == 0)}")

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print("\n=== Sample distribution after SMOTE ===")
print(f"Positive samples: {sum(y_smote == 1)}")
print(f"Negative samples: {sum(y_smote == 0)}")
lr_smote = LogisticRegression()
# model_smote = LogisticRegression(
#     max_iter=1000,
#     random_state=42,
#     class_weight='balanced'
# )  ## 改这个更好
lr_smote.fit(X_smote, y_smote)
y_pred = lr_smote.predict(X_test)
print("\n=== Logistic Regression with SMOTE ===")
print_metrics(y_test, y_pred)
print_top_features(lr_smote, data.columns.drop('loan_status'))
joblib.dump(lr_smote, 'lr_smote_model.pkl')


# ADASYN
print("\n=== Sample distribution before ADASYN ===")
print(f"Positive samples: {sum(y_train == 1)}")
print(f"Negative samples: {sum(y_train == 0)}")

adasyn = ADASYN()
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
print("\n=== Sample distribution after ADASYN ===")
print(f"Positive samples: {sum(y_adasyn == 1)}")
print(f"Negative samples: {sum(y_adasyn == 0)}")
lr_adasyn = LogisticRegression()
# model_adasyn = LogisticRegression(
#     max_iter=1000,
#     random_state=42,
#     class_weight='balanced'
#     # class_weight={0: 1, 1: 5},
# )         ## 改这个更好
lr_adasyn.fit(X_adasyn, y_adasyn)
y_pred = lr_adasyn.predict(X_test)
print("\n=== Logistic Regression with ADASYN ===")
print_metrics(y_test, y_pred)
print_top_features(lr_adasyn, data.columns.drop('loan_status'))
joblib.dump(lr_adasyn, 'lr_adasyn_model.pkl')

# Add visualization and SHAP analysis

def plot_auc_roc(model, X_test, y_test):
    """AUC-ROC cuvrve"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve for Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig("LR_AUC-ROC_curve.png", dpi=300, bbox_inches='tight')
    plt.show()


def shap_analysis(model, X_train, X_test, feature_names):
    """SHAP importance analysis"""
    explainer = shap.Explainer(model.predict_proba, X_train)
    shap_values = explainer(X_test)
    
    # Global feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[:, :, 1], 
        X_test, 
        feature_names=feature_names, 
        plot_type="bar", 
        show=False,
        max_display=10)
    plt.title("SHAP Feature Importance for Logistic Regression")
    plt.tight_layout()
    plt.savefig("LR_shap_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # Single prediction explanation
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[:, :, 1],
        X_test, 
        feature_names=feature_names, 
        show=False,
        max_display=10)
    plt.title("SHAP Summary Plot for Logistic Regression")
    plt.tight_layout()
    plt.savefig("LR_shap_summaryt.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    

print("\n=== Visualization Analysis for Basic Logistic Regression ===")
plot_auc_roc(lr, X_test, y_test)
shap_analysis(lr, X_train, X_test[:10], data.columns.drop('loan_status'))

# plot_auc_roc(lr_l1, X_test, y_test)
# shap_analysis(lr_l1, X_train, X_test[:10], data.columns.drop('loan_status'))

# plot_auc_roc(lr_weighted, X_test, y_test)
# shap_analysis(lr_weighted, X_train, X_test[:10], data.columns.drop('loan_status'))

# plot_auc_roc(lr_smote, X_test, y_test)
# shap_analysis(lr_smote, X_train, X_test[:10], data.columns.drop('loan_status'))

# plot_auc_roc(lr_adasyn, X_test, y_test)
# shap_analysis(lr_adasyn, X_train, X_test[:10], data.columns.drop('loan_status'))
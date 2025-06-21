import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import joblib

# Load data
data = pd.read_csv('process_data.csv')

# Prepare features and target variable
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

def evaluate_model(model, X_train, y_train, X_test, y_test, method_name):
    """
    Function to evaluate model performance
    :param model: Trained model
    :param X_train: Training features
    :param y_train: Training labels
    :param X_test: Test features
    :param y_test: Test labels
    :param method_name: Model method name
    :return: Trained model
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print(f"\n{method_name} evaluation results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

# 2. SMOTE Voting Classifier
print("\n=== SMOTE Voting Classifier ===")
# Apply SMOTE oversampling
smote = SMOTE(
    sampling_strategy=0.8,
    random_state=42,
    k_neighbors=3
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Load pre-trained SMOTE models
logreg_smote = joblib.load('credit_risk_model_smote.pkl')
xgb_smote = joblib.load('xgboost_model_smote.pkl')
lgbm_smote = joblib.load('lightgbm_model_smote.pkl')
rf_smote = joblib.load('random_forest_model_smote.pkl')

voting = VotingClassifier(
    estimators=[
        ('logreg', logreg_smote),
        ('xgb', xgb_smote),
        ('lgbm', lgbm_smote),
        ('rf', rf_smote)
    ],
    voting='soft'
)
voting_smote = evaluate_model(
    voting, X_train_smote, y_train_smote, X_test, y_test,
    "SMOTE Voting Classifier"
)

print("\n============================================================================")
voting_noweight = evaluate_model(
    voting, X_train, y_train, X_test, y_test,
    "noweight"
)


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
data = pd.read_csv('process_data_nolog.csv')

# Prepare features and target variable
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split train and test sets
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
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print(f"\n{method_name} evaluation results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model


# 0. Pure voting classifier (no class weights/resampling)
print("\n=== Pure Voting Classifier ===")
# Load pre-trained models
logreg = joblib.load('lr_smote_model.pkl')
xgb = joblib.load('xgb_adasyn_model.pkl')
lgbm = joblib.load('lgb_adasyn_model.pkl')
rf = joblib.load('rf_smote_model.pkl')
voting_classifier = VotingClassifier(
    estimators=[
        ('logreg', logreg),
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('rf', rf)
    ],
    voting='soft'
)
voting_classifier.fit(X_train, y_train)
voting_pure = evaluate_model(
    voting_classifier, X_train, y_train, X_test, y_test,
    "Pure Voting Classifier"
)
# Save model
joblib.dump(voting_pure, 'voting_pure.pkl')


# # Train with oversampled data, evaluate with original training data
# voting_classifier.fit(X_train_smote, y_train_smote)
# voting_smote = evaluate_model(
#     voting_classifier, X_train, y_train, X_test, y_test,
#     "SMOTE Voting Classifier"
# )

# # Train with oversampled data, evaluate with original training data
# voting_classifier.fit(X_train_adasyn, y_train_adasyn)
# voting_adasyn = evaluate_model(
#     voting_classifier, X_train, y_train, X_test, y_test,
#     "ADASYN Voting Classifier"
# )
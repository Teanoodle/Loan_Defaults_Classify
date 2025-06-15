import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE, ADASYN
import lightgbm as lgb
import joblib

# Data loading
data = pd.read_csv('process_data.csv')

# Feature and target variable
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

def evaluate_model(model, X_train, y_train, X_test, y_test, method_name):
    # train the model
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # eavluate the model
    print(f"\n{method_name}Evaluate results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))
    print("\n Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

# 0. Pure LightGBM method (without class weights/resampling)
print("\n=== Pure LightGBM ===")
model_basic = lgb.LGBMClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    objective='binary',
    metric='auc',
    verbosity=-1,
    min_split_gain=0.01,
    min_data_in_leaf=20
)
model_basic = evaluate_model(
    model_basic, X_train, y_train, X_test, y_test,
    "Pure LightGBM"
)

# 1. Class weights method
print("\n=== LightGBM+class weights ===")
# Calculate scale_pos_weight
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

model_weighted = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    objective='binary',
    metric='auc',
    verbosity=-1,  # banish warnings
    min_split_gain=0.01,  # decide the minimum gain to make a split
    min_data_in_leaf=20  # prevent small leaf nodes
)
model_weighted = evaluate_model(
    model_weighted, X_train, y_train, X_test, y_test, 
    "LightGBM+class weights"
)

# 2. SMOTE
print("\n=== LightGBM+SMOTE ===")
smote = SMOTE(
    sampling_strategy=0.8,
    random_state=42,
    k_neighbors=3
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = lgb.LGBMClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    objective='binary',
    metric='auc',
    verbosity=-1,
    min_split_gain=0.01,
    min_data_in_leaf=20
)
model_smote = evaluate_model(
    model_smote, X_train_smote, y_train_smote, X_test, y_test,
    "LightGBM+SMOTE"
)

# 3. ADASYN
print("\n=== LightGBM+ADASYN ===")
adasyn = ADASYN(
    sampling_strategy=0.8,
    random_state=42,
    n_neighbors=3
)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

model_adasyn = lgb.LGBMClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    objective='binary',
    metric='auc',
    verbosity=-1,
    min_split_gain=0.01,
    min_data_in_leaf=20
)
model_adasyn = evaluate_model(
    model_adasyn, X_train_adasyn, y_train_adasyn, X_test, y_test,
    "LightGBM+ADASYN"
)

# model saving
joblib.dump(model_basic, 'lightgbm_model_basic.pkl')
joblib.dump(model_weighted, 'lightgbm_model_weighted.pkl')
joblib.dump(model_smote, 'lightgbm_model_smote.pkl')
joblib.dump(model_adasyn, 'lightgbm_model_adasyn.pkl')

# # Features importance
# print("\n Important features(top10):")
# importance = pd.Series(model_weighted.feature_importances_, index=X.columns)
# print(importance.sort_values(ascending=False).head(10))
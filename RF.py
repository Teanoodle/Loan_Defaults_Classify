import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
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

# 1. class_weight method
print("\n=== RF+class weights ===")
model_weighted = RandomForestClassifier(
    #class_weight={0: 1, 1: 5},  # 提高少数类的权重
    class_weight='balanced',
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model_weighted = evaluate_model(
    model_weighted, X_train, y_train, X_test, y_test, 
    "RF+class weights"
)

# 2. SMOTE方法
print("\n=== RF+SMOTE ===")
smote = SMOTE(
    sampling_strategy=0.8,
    random_state=42,
    k_neighbors=3
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = RandomForestClassifier(
    # class_weight={0: 1, 1: 5},  # make the minority class more weighted
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model_smote = evaluate_model(
    model_smote, X_train_smote, y_train_smote, X_test, y_test,
    "RF+SMOTE"
)

# 3. ADASYN
print("\n=== RF+ADASYN ===")
adasyn = ADASYN(
    sampling_strategy=0.8,
    random_state=42,
    n_neighbors=3
)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

model_adasyn = RandomForestClassifier(
    # class_weight={0: 1, 1: 5},  # make the minority class more weighted
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model_adasyn = evaluate_model(
    model_adasyn, X_train_adasyn, y_train_adasyn, X_test, y_test,
    "RF+ADASYN"
)

# # 4. SMOTEENN
# print("\n=== RF+SMOTEENN ===")
# smote_enn = SMOTEENN(
#     sampling_strategy=0.8,
#     random_state=42,
#     smote=SMOTE(k_neighbors=3)
# )
# X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train, y_train)

# model_smoteenn = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )
# model_smoteenn = evaluate_model(
#     model_smoteenn, X_train_smoteenn, y_train_smoteenn, X_test, y_test,
#     "RF+SMOTEENN"
# )

# model saving
joblib.dump(model_weighted, 'random_forest_model_weighted.pkl')
# joblib.dump(model_smoteenn, 'random_forest_model_smoteenn.pkl')
joblib.dump(model_smote, 'random_forest_model_smote.pkl')
joblib.dump(model_adasyn, 'random_forest_model_adasyn.pkl')

# # Features importance
# print("\n Important features(top10):")
# importance = pd.Series(model_weighted.feature_importances_, index=X.columns)
# print(importance.sort_values(ascending=False).head(10))
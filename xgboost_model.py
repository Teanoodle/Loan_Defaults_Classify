import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('process_data_nolog.csv')
# data = pd.read_csv('process_data.csv')   ##这个更好


# Feature engineering - XGBoost feature importance selection
X = data.drop('loan_status', axis=1)
y = data['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def print_top_features(model, feature_names):
    """Print top 5 important features with their importance scores"""
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()[::-1]
    print("\nTop 5 important features:")
    for i in range(5):
        print(f"{feature_names[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.6f}")

def print_metrics(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Basic model
print("=== Basic XGBoost ===")
xgb = XGBClassifier()
# xgb = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(xgb, data.drop('loan_status', axis=1).columns)
joblib.dump(xgb, 'xgb_model_basic.pkl')

# Feature selection
selector = SelectFromModel(xgb, threshold='median')
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("\n=== XGBoost with Feature Selection ===")
xgb_selected = XGBClassifier()
# xgb_selected = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_selected.fit(X_train_selected, y_train)
y_pred = xgb_selected.predict(X_test_selected)
print_metrics(y_test, y_pred)
print_top_features(xgb_selected, data.drop('loan_status', axis=1).columns[selector.get_support()])
joblib.dump(xgb_selected, 'xgb_selected_model.pkl')

# Class weight method
print("\n=== XGBoost with Class Weights ===")
xgb_weighted = XGBClassifier(scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1))
# scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
# xgb_weighted = XGBClassifier(
#     scale_pos_weight=scale_pos_weight,
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_weighted.fit(X_train, y_train)
y_pred = xgb_weighted.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(xgb_weighted, data.drop('loan_status', axis=1).columns)
joblib.dump(xgb_weighted, 'xgb_weighted_model.pkl')

# SMOTE method
print("\n=== Sample distribution before SMOTE ===")
print(f"Positive samples: {sum(y_train == 1)}")
print(f"Negative samples: {sum(y_train == 0)}")

smote = SMOTE()
# smote = SMOTE(
#     sampling_strategy=0.8,
#     random_state=42,
#     k_neighbors=3
# )
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print("\n=== Sample distribution after SMOTE ===")
print(f"Positive samples: {sum(y_smote == 1)}")
print(f"Negative samples: {sum(y_smote == 0)}")
xgb_smote = XGBClassifier()
# xgb_smote = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_smote.fit(X_smote, y_smote)
y_pred = xgb_smote.predict(X_test)
print("\n=== XGBoost with SMOTE ===")
print_metrics(y_test, y_pred)
print_top_features(xgb_smote, data.drop('loan_status', axis=1).columns)
joblib.dump(xgb_smote, 'xgb_smote_model.pkl')

# ADASYN method
print("\n=== Sample distribution before ADASYN ===")
print(f"Positive samples: {sum(y_train == 1)}")
print(f"Negative samples: {sum(y_train == 0)}")

adasyn = ADASYN()
# adasyn = ADASYN(
#     sampling_strategy=0.8,
#     random_state=42,
#     n_neighbors=3
# )
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
print("\n=== Sample distribution after ADASYN ===")
print(f"Positive samples: {sum(y_adasyn == 1)}")
print(f"Negative samples: {sum(y_adasyn == 0)}")
xgb_adasyn = XGBClassifier()
# xgb_adasyn = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_adasyn.fit(X_adasyn, y_adasyn)
y_pred = xgb_adasyn.predict(X_test)
print("\n=== XGBoost with ADASYN ===")
print_metrics(y_test, y_pred)
print_top_features(xgb_adasyn, data.drop('loan_status', axis=1).columns)
joblib.dump(xgb_adasyn, 'xgb_adasyn_model.pkl')

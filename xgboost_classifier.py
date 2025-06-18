import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import xgboost as xgb
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

# 特征工程优化
print("\n=== 特征工程优化 ===")

# 1. 增加特征选择数量至15个
selector = xgb.XGBClassifier(random_state=42)
selector.fit(X_train, y_train)
importance = pd.Series(selector.feature_importances_, index=X.columns)
selected_features = importance.sort_values(ascending=False).head(15).index.tolist()
print("Selected features (Top15):", selected_features)

# # 2. 添加特征交互
# if 'total_debt' in X.columns and 'income' in X.columns:
#     # 债务收入比
#     X_train['debt_to_income'] = X_train['total_debt'] / (X_train['income'] + 1e-6)
#     X_test['debt_to_income'] = X_test['total_debt'] / (X_test['income'] + 1e-6)
    
#     # 新增：贷款支付收入比
#     if 'loan_amnt' in X.columns:
#         X_train['payment_to_income'] = X_train['loan_amnt'] / (X_train['income'] + 1e-6)
#         X_test['payment_to_income'] = X_test['loan_amnt'] / (X_test['income'] + 1e-6)
    
#     selected_features.extend(['debt_to_income', 'payment_to_income'])

# 3. 调整类别权重系数 (+20%)
global scale_pos_weight
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) * 1.2

# 应用特征选择
X_train = X_train[selected_features]
X_test = X_test[selected_features]
print(f"最终使用特征数量: {len(selected_features)}")

# 保存特征选择结果供其他模型使用
pd.Series(selected_features).to_csv('selected_features.csv', index=False)

def evaluate_model(model, X_train, y_train, X_test, y_test, method_name):
    # train the model (特征选择已在外部完成)
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

# 0. Pure XGBoost method (without class weights/resampling)
print("\n=== Pure XGBoost method ===")
model_basic = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='auc'
)
model_basic = evaluate_model(
    model_basic, X_train, y_train, X_test, y_test,
    "Pure XGBoost method"
)

# 1. Class weights method
print("\n=== XGBoost+Class weights ===")
# 使用全局调整后的scale_pos_weight
print(f"调整后的类别权重: {scale_pos_weight:.2f}")

model_weighted = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='auc'
)
model_weighted = evaluate_model(
    model_weighted, X_train, y_train, X_test, y_test, 
    "XGBoost+Class weights"
)

# 2. SMOTE
print("\n=== XGBoost+SMOTE ===")
smote = SMOTE(
    sampling_strategy=0.8,
    random_state=42,
    k_neighbors=3
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='auc'
)
model_smote = evaluate_model(
    model_smote, X_train_smote, y_train_smote, X_test, y_test,
    "XGBoost+SMOTE"
)

# 3. ADASYN
print("\n=== XGBoost+ADASYN ===")
adasyn = ADASYN(
    sampling_strategy=0.8,
    random_state=42,
    n_neighbors=3
)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

model_adasyn = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='auc'
)
model_adasyn = evaluate_model(
    model_adasyn, X_train_adasyn, y_train_adasyn, X_test, y_test,
    "XGBoost+ADASYN"
)

# # 4. SMOTEENN
# print("\n=== XGBoost+SMOTEENN ===")
# smote_enn = SMOTEENN(
#     sampling_strategy=0.8,
#     random_state=42,
#     smote=SMOTE(k_neighbors=3)
# )
# X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train, y_train)

# model_smoteenn = xgb.XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
# model_smoteenn = evaluate_model(
#     model_smoteenn, X_train_smoteenn, y_train_smoteenn, X_test, y_test,
#     "XGBoost+SMOTEENN"
# )

# Model saving
joblib.dump(model_basic, 'xgboost_model_basic.pkl')
joblib.dump(model_weighted, 'xgboost_model_weighted.pkl')
joblib.dump(model_smote, 'xgboost_model_smote.pkl')
joblib.dump(model_adasyn, 'xgboost_model_adasyn.pkl')
# joblib.dump(model_smoteenn, 'xgboost_model_smoteenn.pkl')

# 特征重要性可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
pd.Series(model_weighted.feature_importances_, 
         index=X_train.columns).sort_values().plot.barh()
plt.title('Optimized Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\n特征重要性图已保存为feature_importance.png")

# 显示最重要的5个特征
print("\nTop 5重要特征:")
print(pd.Series(model_weighted.feature_importances_, 
               index=X_train.columns).sort_values(ascending=False).head(5))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb
import joblib

# 加载数据
data = pd.read_csv('cleaned_credit_risk_dataset_processed.csv')

# 准备特征和目标变量
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

def evaluate_model(model, X_train, y_train, X_test, y_test, method_name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估模型
    print(f"\n{method_name}评估结果:")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("召回率:", recall_score(y_test, y_pred))
    print("F1分数:", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

# 1. 类权重方法
print("\n=== XGBoost类权重方法 ===")
# 计算正负样本比例
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

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
    "XGBoost类权重方法"
)

# 2. SMOTE方法
print("\n=== XGBoost+SMOTE方法 ===")
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
    "XGBoost+SMOTE方法"
)

# 3. ADASYN方法
print("\n=== XGBoost+ADASYN方法 ===")
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
    "XGBoost+ADASYN方法"
)

# 保存模型
joblib.dump(model_weighted, 'xgboost_model_weighted.pkl')
joblib.dump(model_smote, 'xgboost_model_smote.pkl')
joblib.dump(model_adasyn, 'xgboost_model_adasyn.pkl')

# 特征重要性分析
print("\n特征重要性(top10):")
importance = pd.Series(model_weighted.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(10))
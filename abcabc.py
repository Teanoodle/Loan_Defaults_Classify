import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import joblib

# 加载数据
data = pd.read_csv('process_data.csv')

# 准备特征和目标变量
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

def evaluate_model(model, X_train, y_train, X_test, y_test, method_name):
    """
    评估模型性能的函数
    :param model: 训练好的模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param X_test: 测试特征
    :param y_test: 测试标签
    :param method_name: 模型方法名称
    :return: 训练好的模型
    """
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

# 2. SMOTE方法投票分类器
print("\n=== SMOTE方法投票分类器 ===")
# 应用SMOTE过采样
smote = SMOTE(
    sampling_strategy=0.8,
    random_state=42,
    k_neighbors=3
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 加载预训练的SMOTE模型
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
    "SMOTE方法投票分类器"
)

print("\n============================================================================")
voting_noweight = evaluate_model(
    voting, X_train, y_train, X_test, y_test,
    "noweight"
)


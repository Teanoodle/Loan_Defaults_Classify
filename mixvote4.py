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
data = pd.read_csv('process_data_nolog.csv')

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


# 0. 纯投票分类器（无类权重/重采样）
print("\n=== 纯投票分类器 ===")
# 加载预训练的模型
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
voting_pure = evaluate_model(
    voting_classifier, X_train, y_train, X_test, y_test,
    "纯投票分类器"
)
# 保存模型
joblib.dump(voting_pure, 'voting_pure.pkl')


# # 训练使用过采样数据，评估使用原始训练数据
# voting_classifier.fit(X_train_smote, y_train_smote)
# voting_smote = evaluate_model(
#     voting_classifier, X_train, y_train, X_test, y_test,
#     "SMOTE方法投票分类器"
# )

# # 训练使用过采样数据，评估使用原始训练数据
# voting_classifier.fit(X_train_adasyn, y_train_adasyn)
# voting_adasyn = evaluate_model(
#     voting_classifier, X_train, y_train, X_test, y_test,
#     "ADASYN方法投票分类器"
# )
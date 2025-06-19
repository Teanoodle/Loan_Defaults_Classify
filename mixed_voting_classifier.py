import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.ensemble import VotingClassifier
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

def evaluate_model(model, X_test, y_test, method_name):
    """
    评估模型性能的函数
    :param model: 训练好的模型
    :param X_test: 测试特征
    :param y_test: 测试标签
    :param method_name: 模型方法名称
    :return: 训练好的模型
    """
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

# 创建混合投票分类器
print("\n=== 混合方法投票分类器 ===")
# 加载预训练的不同处理方法模型
logreg_weighted = joblib.load('credit_risk_model_adasyn.pkl')  # 类权重
xgb_adasyn = joblib.load('xgboost_model_basic.pkl')            # ADASYN
lgbm_smote = joblib.load('lightgbm_model_smote.pkl')           # SMOTE
rf_basic = joblib.load('random_forest_model_basic.pkl')        # 基础方法

mixed_voting = VotingClassifier(
    estimators=[
        ('logreg_weighted', logreg_weighted),
        ('xgb_adasyn', xgb_adasyn),
        ('lgbm_smote', lgbm_smote),
        ('rf_basic', rf_basic)
        
    ],
    voting='soft',  # 使用概率投票
    weights=[1.8, 1,1.2,1],  # 为想要的模型赋予更高权重
    n_jobs=-1      # 使用所有CPU核心
)

# 虽然基模型已预训练，但VotingClassifier仍需fit()
# 使用原始训练数据拟合（不重新训练基模型）
mixed_voting.fit(X_train, y_train)

# 评估混合模型
mixed_voting = evaluate_model(
    mixed_voting, X_test, y_test,
    "混合方法投票分类器"
)

# # 保存模型
# joblib.dump(mixed_voting, 'mixed_voting_model.pkl')

print("\n混合投票分类器创建完成！")
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

# 1. 类权重方法投票分类器
print("\n=== 类权重方法投票分类器 ===")
# 加载预训练的类权重模型
logreg_weighted = joblib.load('credit_risk_model_weighted.pkl')
xgb_weighted = joblib.load('xgboost_model_weighted.pkl')
lgbm_weighted = joblib.load('lightgbm_model_weighted.pkl')
rf_weighted = joblib.load('random_forest_model_weighted.pkl')

voting_weighted = VotingClassifier(
    estimators=[
        ('logreg', logreg_weighted),
        ('xgb', xgb_weighted),
        ('lgbm', lgbm_weighted),
        ('rf', rf_weighted)
    ],
    voting='soft'
)

voting_weighted = evaluate_model(
    voting_weighted, X_train, y_train, X_test, y_test,
    "类权重方法投票分类器"
)

# 保存模型
joblib.dump(voting_weighted, 'voting_model_weighted.pkl')

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

voting_smote = VotingClassifier(
    estimators=[
        ('logreg', logreg_smote),
        ('xgb', xgb_smote),
        ('lgbm', lgbm_smote),
        ('rf', rf_smote)
    ],
    voting='soft'
)

voting_smote = evaluate_model(
    voting_smote, X_train_smote, y_train_smote, X_test, y_test,
    "SMOTE方法投票分类器"
)

# 保存模型
joblib.dump(voting_smote, 'voting_model_smote.pkl')

# 3. ADASYN方法投票分类器
print("\n=== ADASYN方法投票分类器 ===")
# 应用ADASYN过采样
adasyn = ADASYN(
    sampling_strategy=0.8,
    random_state=42,
    n_neighbors=3
)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# 加载预训练的ADASYN模型
logreg_adasyn = joblib.load('credit_risk_model_adasyn.pkl')
xgb_adasyn = joblib.load('xgboost_model_adasyn.pkl')
lgbm_adasyn = joblib.load('lightgbm_model_adasyn.pkl')
rf_adasyn = joblib.load('random_forest_model_adasyn.pkl')

voting_adasyn = VotingClassifier(
    estimators=[
        ('logreg', logreg_adasyn),
        ('xgb', xgb_adasyn),
        ('lgbm', lgbm_adasyn),
        ('rf', rf_adasyn)
    ],
    voting='soft'
)

voting_adasyn = evaluate_model(
    voting_adasyn, X_train_adasyn, y_train_adasyn, X_test, y_test,
    "ADASYN方法投票分类器"
)

# 保存模型
joblib.dump(voting_adasyn, 'voting_model_adasyn.pkl')

# 特征重要性分析
print("\n特征重要性分析(top10):")
for name, model in voting_adasyn.named_estimators_.items():
    try:
        importance = pd.Series(model.feature_importances_, index=X.columns)
        print(f"\n{name}特征重要性:")
        print(importance.sort_values(ascending=False).head(10))
    except AttributeError:
        print(f"\n{name}不支持特征重要性分析")
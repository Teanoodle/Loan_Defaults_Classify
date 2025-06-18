import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb
import lightgbm as lgb
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

# 定义基学习器（新增随机森林）
estimators = [
    ('lr', LogisticRegression(max_iter=5000)),
    ('xgb', xgb.XGBClassifier(max_depth=5, random_state=42)),
    ('lgb', lgb.LGBMClassifier(
        max_depth=5,
        random_state=42,
        verbosity=-1,
        min_split_gain=0.01,
        min_data_in_leaf=20
    )),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        # class_weight='balanced',   # 提高少数类的权重
        class_weight={0: 1, 1: 5},
        random_state=42
    ))
]

# 1. 类权重方法
print("\n=== Stacking类权重方法 ===")
scale_pos_weight = 5

# 创建带权重的基学习器（新增随机森林）
weighted_estimators = [
    ('lr', LogisticRegression(class_weight={0: 1, 1: 5}, max_iter=5000)),
    ('xgb', xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=5,
        random_state=42
    )),
    ('lgb', lgb.LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=5,
        random_state=42,
        verbosity=-1,
        min_split_gain=0.01,
        min_data_in_leaf=20
    )),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        # class_weight='balanced',
        class_weight={0: 1, 1: 5},  # 提高少数类的权重
        random_state=42
    ))
]

stacking_weighted = StackingClassifier(
    estimators=weighted_estimators,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    stack_method='predict_proba'
)

stacking_weighted = evaluate_model(
    stacking_weighted, X_train, y_train, X_test, y_test,
    "Stacking类权重方法"
)

# 2. SMOTE方法
print("\n=== Stacking+SMOTE方法 ===")
smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

stacking_smote = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'
)
stacking_smote = evaluate_model(
    stacking_smote, X_train_smote, y_train_smote, X_test, y_test,
    "Stacking+SMOTE方法"
)

# 3. ADASYN方法
print("\n=== Stacking+ADASYN方法 ===")
adasyn = ADASYN(sampling_strategy=0.8, random_state=42, n_neighbors=3)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

stacking_adasyn = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'
)
stacking_adasyn = evaluate_model(
    stacking_adasyn, X_train_adasyn, y_train_adasyn, X_test, y_test,
    "Stacking+ADASYN方法"
)

# 保存模型
joblib.dump(stacking_weighted, 'stacking_model_weighted.pkl')
joblib.dump(stacking_smote, 'stacking_model_smote.pkl')
joblib.dump(stacking_adasyn, 'stacking_model_adasyn.pkl')

# 创建并评估VotingClassifier
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('weighted', stacking_weighted),
        ('smote', stacking_smote),
        ('adasyn', stacking_adasyn)
    ],
    voting='soft',
    weights=[1.5, 0.8, 1]  # 为想要的模型赋予更高权重
)

print("\n=== VotingClassifier集成三种方法 ===")
voting_clf = evaluate_model(
    voting_clf, X_train, y_train, X_test, y_test,
    "VotingClassifier(加权+SMOTE+ADASYN)"
)

# 保存最终投票模型
joblib.dump(voting_clf, 'final_voting_model.pkl')
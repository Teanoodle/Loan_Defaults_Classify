import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from logistic_regression import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# 读取数据
data = pd.read_csv('process_data_nolog.csv')

# 特征工程 - 使用与各模型相同的预处理
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def print_metrics(y_true, y_pred):
    print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
    print(f"召回率: {recall_score(y_true, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(y_true, y_pred))
    print("分类报告:")
    print(classification_report(y_true, y_pred))


# 初始化各模型（使用最佳参数）
models = [
    ('lr', LogisticRegression(class_weight='balanced')),
    # ('lr' , joblib.load('credit_risk_model_basic.pkl')),
    ('xgb', XGBClassifier(scale_pos_weight=sum(y==0)/sum(y==1))),
    # ('xgb' , joblib.load('xgboost_model_basic.pkl')),
    ('lgb', LGBMClassifier(is_unbalance=True)),
    # ('lgb', joblib.load('lightgbm_model_smote.pkl')),
    ('rf', RandomForestClassifier(class_weight='balanced'))
    # ('rf' , joblib.load('random_forest_model_basic.pkl'))
]

# 创建投票分类器
print("=== 硬投票集成 ===")
voting_hard = VotingClassifier(estimators=models, voting='hard')
voting_hard.fit(X_train, y_train)
y_pred = voting_hard.predict(X_test)
print_metrics(y_test, y_pred)

print("\n=== 软投票集成 ===")
voting_soft = VotingClassifier(estimators=models, voting='soft')
voting_soft.fit(X_train, y_train)
y_pred = voting_soft.predict(X_test)
print_metrics(y_test, y_pred)

# 加权投票（根据各模型AUC-ROC表现加权）
print("\n=== 加权投票集成 ===")
weights = [0.85, 0.90, 0.92, 0.88]  # 假设的权重
voting_weighted = VotingClassifier(estimators=models, voting='soft', weights=weights)
voting_weighted.fit(X_train, y_train)
y_pred = voting_weighted.predict(X_test)
print_metrics(y_test, y_pred)
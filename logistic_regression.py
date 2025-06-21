import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np


# 读取数据
data = pd.read_csv('process_data_nolog.csv')
# data = pd.read_csv('process_data.csv')  # 用这个更好

# 特征工程 - 逻辑回归需要标准化
X = data.drop('loan_status', axis=1)
y = data['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def print_metrics(y_true, y_pred):
    print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
    print(f"召回率: {recall_score(y_true, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(y_true, y_pred))
    print("分类报告:")
    print(classification_report(y_true, y_pred))

def print_top_features(model, feature_names, n=5):
    """Print top n features with their coefficients"""
    coef = model.coef_[0]
    indices = np.argsort(np.abs(coef))[::-1][:n]
    print("\nTop features:")
    for i in indices:
        print(f"{feature_names[i]}: {coef[i]:.4f}")


# 基础模型
print("=== 基础逻辑回归 ===")
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lr, data.columns.drop('loan_status'))
joblib.dump(lr, 'lr_model_basic.pkl')



# L1正则方法
print("\n=== L1正则后的逻辑回归 ===")
lr_l1 = LogisticRegression(
    penalty='l1', 
    # penalty='elasticnet',  # Combine L1 and L2 regularization
    # l1_ratio=0.5,  # Balance between L1 and L2
    solver='saga',  # Better for large datasets
    C=0.1
)
lr_l1.fit(X_train, y_train)
y_pred = lr_l1.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lr_l1, data.columns.drop('loan_status'))
joblib.dump(lr_l1, 'lr_l1_model.pkl')



# 类权重方法
print("\n=== 带类权重的逻辑回归 ===")
lr_weighted = LogisticRegression(class_weight='balanced')
lr_weighted.fit(X_train, y_train)
y_pred = lr_weighted.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lr_weighted, data.columns.drop('loan_status'))
joblib.dump(lr_weighted, 'lr_weighted_model.pkl')


# SMOTE方法
print("\n=== SMOTE处理前的样本分布 ===")
print(f"正类样本数: {sum(y_train == 1)}")
print(f"负类样本数: {sum(y_train == 0)}")

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print("\n=== SMOTE处理后的样本分布 ===")
print(f"正类样本数: {sum(y_smote == 1)}")
print(f"负类样本数: {sum(y_smote == 0)}")
lr_smote = LogisticRegression()
# model_smote = LogisticRegression(
#     max_iter=1000,
#     random_state=42,
#     class_weight='balanced'
# )  ## 改这个更好
lr_smote.fit(X_smote, y_smote)
y_pred = lr_smote.predict(X_test)
print("\n=== SMOTE处理后的逻辑回归 ===")
print_metrics(y_test, y_pred)
print_top_features(lr_smote, data.columns.drop('loan_status'))
joblib.dump(lr_smote, 'lr_smote_model.pkl')


# ADASYN方法
print("\n=== ADASYN处理前的样本分布 ===")
print(f"正类样本数: {sum(y_train == 1)}")
print(f"负类样本数: {sum(y_train == 0)}")

adasyn = ADASYN()
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
print("\n=== ADASYN处理后的样本分布 ===")
print(f"正类样本数: {sum(y_adasyn == 1)}")
print(f"负类样本数: {sum(y_adasyn == 0)}")
lr_adasyn = LogisticRegression()
# model_adasyn = LogisticRegression(
#     max_iter=1000,
#     random_state=42,
#     class_weight='balanced'
#     # class_weight={0: 1, 1: 5},
# )         ## 改这个更好
lr_adasyn.fit(X_adasyn, y_adasyn)
y_pred = lr_adasyn.predict(X_test)
print("\n=== ADASYN处理后的逻辑回归 ===")
print_metrics(y_test, y_pred)
print_top_features(lr_adasyn, data.columns.drop('loan_status'))
joblib.dump(lr_adasyn, 'lr_adasyn_model.pkl')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
import joblib

# 加载数据
data = pd.read_csv('cleaned_credit_risk_dataset_processed.csv')

# 准备特征和目标变量
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# 递归特征消除(RFE) - 保留10个最重要特征
print("\n=== 特征选择(RFE) ===")
estimator = LogisticRegression(max_iter=5000, random_state=42)
selector = RFE(estimator, n_features_to_select=10, step=1)
X_selected = selector.fit_transform(X, y)

# 获取被选中的特征名称
selected_features = X.columns[selector.support_]
print("保留的特征:", list(selected_features))
X = X[selected_features]

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
print("\n=== 类权重方法(RFE) ===")
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

model_weighted = LogisticRegression(
    class_weight=class_weights,
    max_iter=5000,
    random_state=42
)
model_weighted = evaluate_model(
    model_weighted, X_train, y_train, X_test, y_test, 
    "类权重方法(RFE)"
)

# 2. SMOTE方法
print("\n=== SMOTE优化方法(RFE) ===")
smote = SMOTE(
    sampling_strategy=0.8,
    random_state=42,
    k_neighbors=3
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression(
    max_iter=5000,
    random_state=42,
    class_weight='balanced'
)
model_smote = evaluate_model(
    model_smote, X_train_smote, y_train_smote, X_test, y_test,
    "SMOTE优化方法(RFE)"
)

# 3. ADASYN方法
print("\n=== ADASYN方法(RFE) ===")
adasyn = ADASYN(
    sampling_strategy=0.8,
    random_state=42,
    n_neighbors=3
)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

model_adasyn = LogisticRegression(
    max_iter=5000,
    random_state=42,
    class_weight='balanced'
)
model_adasyn = evaluate_model(
    model_adasyn, X_train_adasyn, y_train_adasyn, X_test, y_test,
    "ADASYN方法(RFE)"
)

# 保存模型和特征选择器
joblib.dump(model_weighted, 'credit_risk_model_weighted_rfe.pkl')
joblib.dump(model_smote, 'credit_risk_model_smote_rfe.pkl')
joblib.dump(model_adasyn, 'credit_risk_model_adasyn_rfe.pkl')
joblib.dump(selector, 'feature_selector_rfe.pkl')

# 保存选中的特征列表
pd.Series(selected_features).to_csv('selected_features_rfe.csv', index=False)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve, auc
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 加载数据
data = pd.read_csv('process_data_nolog.csv')
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 加载所有模型
models = {
    'LR Basic': joblib.load('lr_model_basic.pkl'),
    'LR Class Weights': joblib.load('lr_weighted_model.pkl'), 
    'LR SMOTE': joblib.load('lr_smote_model.pkl'),
    'LR ADASYN': joblib.load('lr_adasyn_model.pkl'),
    'XGB Basic': joblib.load('xgb_model_basic.pkl'),
    'XGB Weighted': joblib.load('xgb_weighted_model.pkl'),
    'XGB SMOTE': joblib.load('xgb_smote_model.pkl'),
    'XGB ADASYN': joblib.load('xgb_adasyn_model.pkl'),
    'LGB Basic': joblib.load('lgb_model_basic.pkl'),
    'LGB Weighted': joblib.load('lgb_weighted_model.pkl'),
    'LGB SMOTE': joblib.load('lgb_smote_model.pkl'),
    'LGB ADASYN': joblib.load('lgb_adasyn_model.pkl'),
    'RF Basic': joblib.load('rf_model_basic.pkl'),
    'RF Weighted': joblib.load('rf_weighted_model.pkl'),
    'RF SMOTE': joblib.load('rf_smote_model.pkl'),
    'RF ADASYN': joblib.load('rf_adasyn_model.pkl')
}

# 按模型类型分组
model_groups = {
    'Logistic Regression': ['LR Basic', 'LR Class Weights', 'LR SMOTE', 'LR ADASYN'],
    'XGBoost': ['XGB Basic', 'XGB Weighted', 'XGB SMOTE', 'XGB ADASYN'],
    'LightGBM': ['LGB Basic', 'LGB Weighted', 'LGB SMOTE', 'LGB ADASYN'],
    'Random Forest': ['RF Basic', 'RF Weighted', 'RF SMOTE', 'RF ADASYN']
}

# 1. Recall对比折线图（按算法分开）
for group, model_list in model_groups.items():
    plt.figure(figsize=(12, 6))
    recall_scores = []
    for name in model_list:
        model = models[name]
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        recall_scores.append(recall)
    
    plt.plot(model_list, recall_scores, marker='o', markersize=8, linestyle='-', linewidth=2)
    plt.title(f'Recall Score Comparison - {group}')
    plt.ylabel('Recall Score')
    plt.xlabel('Model Type')
    plt.xticks(rotation=45)
    for i, v in enumerate(recall_scores):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'recall_comparison_{group.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. F1 Score对比折线图（按算法分开）
for group, model_list in model_groups.items():
    plt.figure(figsize=(12, 6))
    f1_scores = []
    for name in model_list:
        model = models[name]
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    
    plt.plot(model_list, f1_scores, marker='o', markersize=8, linestyle='-', linewidth=2)
    plt.title(f'F1 Score Comparison - {group}')
    plt.ylabel('F1 Score')
    plt.xlabel('Model Type')
    plt.xticks(rotation=45)
    for i, v in enumerate(f1_scores):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'f1_comparison_{group.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. AUC-ROC曲线对比图（按算法分开）
for group, model_list in model_groups.items():
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple']
    for name, color in zip(model_list, colors):
        model = models[name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison - {group}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'roc_comparison_{group.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
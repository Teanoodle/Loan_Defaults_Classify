import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve, auc, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# load data
# data = pd.read_csv('process_data_nolog.csv')
data = pd.read_csv('process_data.csv')
X = data.drop('loan_status', axis=1)
y = data['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# load models
# Note: Ensure that the models are saved in the same directory as this script or provide the
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
    'RF ADASYN': joblib.load('rf_adasyn_model.pkl'),
    'Voting Classifier': joblib.load('voting.pkl')
}

# class the models into groups for easier plotting
model_groups = {
    'Logistic Regression': ['LR Basic', 'LR Class Weights', 'LR SMOTE', 'LR ADASYN'],
    'XGBoost': ['XGB Basic', 'XGB Weighted', 'XGB SMOTE', 'XGB ADASYN'],
    'LightGBM': ['LGB Basic', 'LGB Weighted', 'LGB SMOTE', 'LGB ADASYN'],
    'Random Forest': ['RF Basic', 'RF Weighted', 'RF SMOTE', 'RF ADASYN'],
    'Ensemble': ['Voting Classifier']
}

# 1. Recall comparison line chart (by algorithm)
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

# 2. F1 Score comparison line chart (by algorithm)
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






# 7. Accracy and Recall comparison line chart (by sampling method)
plt.figure(figsize=(20, 8))
x_labels = ['Basic', 'Class Weighted', 'SMOTE', 'ADASYN']
group_colors = {
    'Logistic Regression': 'blue',
    'XGBoost': 'green',
    'LightGBM': 'red',
    'Random Forest': 'purple',
    'Ensemble': 'gold'
}
markers = ['o', 's', '^', 'D', '*']

# Left - Accuracy
plt.subplot(1, 2, 1)
for group, model_list in model_groups.items():
    accuracy_scores = []
    for name in model_list:
        model = models[name]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
    
    if group == 'Ensemble':
        # Special handling for Ensemble model
        plt.scatter(1, accuracy_scores[0], marker='*', s=200, 
                   color=group_colors[group], label=group, zorder=5)
        plt.text(1, accuracy_scores[0], f"{accuracy_scores[0]:.3f}", 
                ha='center', va='bottom')
    else:
        plt.plot(x_labels, accuracy_scores, marker='o', markersize=8, 
               linestyle='-', linewidth=2, color=group_colors[group], label=group)
        for i, v in enumerate(accuracy_scores):
            plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

plt.title('Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.xlabel('Sampling Method')
plt.legend()
plt.grid(True)

# Right - Recall
plt.subplot(1, 2, 2)
for group, model_list in model_groups.items():
    recall_scores = []
    for name in model_list:
        model = models[name]
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        recall_scores.append(recall)
    
    if group == 'Ensemble':
        # Special handling for Ensemble model
        plt.scatter(1, recall_scores[0], marker='*', s=200, 
                   color=group_colors[group], label=group, zorder=5)
        plt.text(1, recall_scores[0], f"{recall_scores[0]:.3f}", 
                ha='center', va='bottom')
    else:
        plt.plot(x_labels, recall_scores, marker='o', markersize=8, 
               linestyle='-', linewidth=2, color=group_colors[group], label=group)
        for i, v in enumerate(recall_scores):
            plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

plt.title('Recall Score Comparison')
plt.ylabel('Recall Score')
plt.xlabel('Sampling Method')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('accuracy_recall_comparison.png', dpi=300, bbox_inches='tight')
plt.show()





# 8. F1 and ROC comparison line chart (by sampling method)
plt.figure(figsize=(20, 8))

# Left - F1 Score
plt.subplot(1, 2, 1)
for group, model_list in model_groups.items():
    f1_scores = []
    for name in model_list:
        model = models[name]
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    
    if group == 'Ensemble':
        # Special handling for Ensemble model
        plt.scatter(1, f1_scores[0], marker='*', s=200, 
                   color=group_colors[group], label=group, zorder=5)
        plt.text(1, f1_scores[0], f"{f1_scores[0]:.3f}", 
                ha='center', va='bottom')
    else:
        plt.plot(x_labels, f1_scores, marker='o', markersize=8, 
               linestyle='-', linewidth=2, color=group_colors[group], label=group)
        for i, v in enumerate(f1_scores):
            plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

plt.title('F1 Score Comparison')
plt.ylabel('F1 Score')
plt.xlabel('Sampling Method')
plt.legend()
plt.grid(True)

# Right - ROC曲线
plt.subplot(1, 2, 2)
for group, model_list in model_groups.items():
    for name in model_list:
        model = models[name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        if group == 'Ensemble':
            plt.plot(fpr, tpr, color=group_colors[group], lw=3,
                    label=f'{group} (AUC = {roc_auc:.2f})', zorder=5)
        else:
            plt.plot(fpr, tpr, color=group_colors[group], lw=2,
                    label=f'{group} {name.split()[-1]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.savefig('f1_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
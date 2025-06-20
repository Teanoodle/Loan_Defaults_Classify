import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel

# 读取数据
data = pd.read_csv('process_data_nolog.csv')
# data = pd.read_csv('process_data.csv')   ##这个更好


# 特征工程 - XGBoost特征重要性选择
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

# 基础模型
print("=== 基础XGBoost ===")
xgb = XGBClassifier()
# xgb = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(xgb, 'xgb_model_basic.pkl')

# 特征选择
selector = SelectFromModel(xgb, threshold='median')
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("\n=== 特征选择后的XGBoost ===")
xgb_selected = XGBClassifier()
# xgb_selected = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_selected.fit(X_train_selected, y_train)
y_pred = xgb_selected.predict(X_test_selected)
print_metrics(y_test, y_pred)
joblib.dump(xgb_selected, 'xgb_selected_model.pkl')

# 类权重方法
print("\n=== 带类权重的XGBoost ===")
xgb_weighted = XGBClassifier(scale_pos_weight=sum(y==0)/sum(y==1))
# scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
# xgb_weighted = XGBClassifier(
#     scale_pos_weight=scale_pos_weight,
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_weighted.fit(X_train, y_train)
y_pred = xgb_weighted.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(xgb_weighted, 'xgb_weighted_model.pkl')

# SMOTE方法
print("\n=== SMOTE处理后的XGBoost ===")
smote = SMOTE()
# smote = SMOTE(
#     sampling_strategy=0.8,
#     random_state=42,
#     k_neighbors=3
# )
X_smote, y_smote = smote.fit_resample(X_train, y_train)
xgb_smote = XGBClassifier()
# xgb_smote = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_smote.fit(X_smote, y_smote)
y_pred = xgb_smote.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(xgb_smote, 'xgb_smote_model.pkl')

# ADASYN方法
print("\n=== ADASYN处理后的XGBoost ===")
adasyn = ADASYN()
# adasyn = ADASYN(
#     sampling_strategy=0.8,
#     random_state=42,
#     n_neighbors=3
# )
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
xgb_adasyn = XGBClassifier()
# xgb_adasyn = XGBClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='auc'
# )
xgb_adasyn.fit(X_adasyn, y_adasyn)
y_pred = xgb_adasyn.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(xgb_adasyn, 'xgb_adasyn_model.pkl')

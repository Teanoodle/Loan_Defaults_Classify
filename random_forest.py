import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel

# 读取数据
data = pd.read_csv('process_data_nolog.csv')
# data = pd.read_csv('process_data.csv')    ## 用这个更好

# 特征工程 - 随机森林特征重要性选择
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
print("=== 基础随机森林 ===")
rf = RandomForestClassifier()
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(rf, 'rf_model_basic.pkl')

# 特征选择
selector = SelectFromModel(rf, threshold='median')
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("\n=== 特征选择后的随机森林 ===")
rf_selected = RandomForestClassifier()
# rf_selected = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )
rf_selected.fit(X_train_selected, y_train)
y_pred = rf_selected.predict(X_test_selected)
print_metrics(y_test, y_pred)
joblib.dump(rf_selected, 'rf_selected_model.pkl')

# 类权重方法
print("\n=== 带类权重的随机森林 ===")
rf_weighted = RandomForestClassifier(class_weight='balanced')
# rf_weighted = RandomForestClassifier(
#     #class_weight={0: 1, 1: 5},  # 提高少数类的权重
#     class_weight='balanced_subsample',
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )
rf_weighted.fit(X_train, y_train)
y_pred = rf_weighted.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(rf_weighted, 'rf_weighted_model.pkl')

# SMOTE方法
print("\n=== SMOTE处理后的随机森林 ===")
smote = SMOTE()
# smote = SMOTE(
#     sampling_strategy=0.8,
#     random_state=42,
#     k_neighbors=3
# )
X_smote, y_smote = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier()
# rf_smote = RandomForestClassifier(
#     # class_weight={0: 1, 1: 5},  # make the minority class more weighted
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )
rf_smote.fit(X_smote, y_smote)
y_pred = rf_smote.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(rf_smote, 'rf_smote_model.pkl')

# ADASYN方法
print("\n=== ADASYN处理后的随机森林 ===")
adasyn = ADASYN()
# adasyn = ADASYN(
#     sampling_strategy=0.8,
#     random_state=42,
#     n_neighbors=3
# )
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
rf_adasyn = RandomForestClassifier()
# rf_adasyn = RandomForestClassifier(
#     # class_weight={0: 1, 1: 5},  # make the minority class more weighted
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )
rf_adasyn.fit(X_adasyn, y_adasyn)
y_pred = rf_adasyn.predict(X_test)
print_metrics(y_test, y_pred)
joblib.dump(rf_adasyn, 'rf_adasyn_model.pkl')
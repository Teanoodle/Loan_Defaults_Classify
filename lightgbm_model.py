import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv('process_data_nolog.csv')
# data = pd.read_csv('process_data.csv')

# 特征工程 - LightGBM特征重要性选择
X = data.drop('loan_status', axis=1)
y = data['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def print_top_features(model, feature_names):
    """Print top 5 important features with their importance scores"""
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()[::-1]
    print("\nTop 5 important features:")
    for i in range(5):
        print(f"{feature_names[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.6f}")

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
print("=== 基础LightGBM ===")
lgb_model = lgb.LGBMClassifier()
# lgb_model = lgb.LGBMClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     objective='binary',
#     metric='auc',
#     verbosity=-1,
#     min_split_gain=0.01,
#     min_data_in_leaf=20
# )
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lgb_model, data.drop('loan_status', axis=1).columns)
joblib.dump(lgb_model, 'lgb_model_basic.pkl')

# 特征选择
selector = SelectFromModel(lgb_model, threshold='median')
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("\n=== 特征选择后的LightGBM ===")
lgb_selected = lgb.LGBMClassifier()
# lgb_selected = lgb.LGBMClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     objective='binary',
#     metric='auc',
#     verbosity=-1,
#     min_split_gain=0.01,
#     min_data_in_leaf=20
# )
lgb_selected.fit(X_train_selected, y_train)
y_pred = lgb_selected.predict(X_test_selected)
print_metrics(y_test, y_pred)
print_top_features(lgb_selected, data.drop('loan_status', axis=1).columns[selector.get_support()])
joblib.dump(lgb_selected, 'lgb_selected_model.pkl')

# 类权重方法
print("\n=== 带类权重的LightGBM ===")
lgb_weighted = lgb.LGBMClassifier(is_unbalance=True)
# scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
# lgb_weighted = lgb.LGBMClassifier(
#     scale_pos_weight=scale_pos_weight,
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     objective='binary',
#     metric='auc',
#     verbosity=-1,  # banish warnings
#     min_split_gain=0.01,  # decide the minimum gain to make a split
#     min_data_in_leaf=20  # prevent small leaf nodes
# )
lgb_weighted.fit(X_train, y_train)
y_pred = lgb_weighted.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lgb_weighted, data.drop('loan_status', axis=1).columns)
joblib.dump(lgb_weighted, 'lgb_weighted_model.pkl')

# SMOTE方法
print("\n=== SMOTE处理后的LightGBM ===")
smote = SMOTE()
# smote = SMOTE(
#     sampling_strategy=0.8,
#     random_state=42,
#     k_neighbors=3
# )
X_smote, y_smote = smote.fit_resample(X_train, y_train)
lgb_smote = lgb.LGBMClassifier()
# lgb_smote = lgb.LGBMClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     objective='binary',
#     metric='auc',
#     verbosity=-1,
#     min_split_gain=0.01,
#     min_data_in_leaf=20
# )
lgb_smote.fit(X_smote, y_smote)
y_pred = lgb_smote.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lgb_smote, data.drop('loan_status', axis=1).columns)
joblib.dump(lgb_smote, 'lgb_smote_model.pkl')

# ADASYN方法
print("\n=== ADASYN处理后的LightGBM ===")
adasyn = ADASYN()
# adasyn = ADASYN(
#     sampling_strategy=0.8,
#     random_state=42,
#     n_neighbors=3
# )
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
lgb_adasyn = lgb.LGBMClassifier()
# lgb_adasyn = lgb.LGBMClassifier(
#     max_depth=5,
#     learning_rate=0.1,
#     n_estimators=100,
#     random_state=42,
#     objective='binary',
#     metric='auc',
#     verbosity=-1,
#     min_split_gain=0.01,
#     min_data_in_leaf=20
# )
lgb_adasyn.fit(X_adasyn, y_adasyn)
y_pred = lgb_adasyn.predict(X_test)
print_metrics(y_test, y_pred)
print_top_features(lgb_adasyn, data.drop('loan_status', axis=1).columns)
joblib.dump(lgb_adasyn, 'lgb_adasyn_model.pkl')
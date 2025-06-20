import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

def xgb_feature_importance(X, y, top_n=15):
    """使用XGBoost获取特征重要性"""
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(top_n).index.tolist()

def create_xgb_features(X):
    """创建XGBoost特有的特征"""
    X_new = X.copy()
    # 添加债务收入比
    if 'total_debt' in X.columns and 'income' in X.columns:
        X_new['debt_to_income'] = X['total_debt'] / (X['income'] + 1e-6)
    # 添加贷款支付收入比
    if 'loan_amnt' in X.columns and 'income' in X.columns:
        X_new['payment_to_income'] = X['loan_amnt'] / (X['income'] + 1e-6)
    return X_new


# ============================================================================================================================


def prepare_xgb_features(file_path='process_data.csv'):
    """XGBoost特征工程流程"""
    # 加载数据
    data = pd.read_csv(file_path)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    # 特征工程
    print("\n=== XGBoost特征工程 ===")
    # 1. 获取重要特征
    selected_features = xgb_feature_importance(X, y)
    print(f"XGBoost选择的重要特征 ({len(selected_features)}个):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 2. 创建特有特征
    X = create_xgb_features(X)
    
    # 3. 应用特征选择
    X = X[selected_features]
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n最终特征维度: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler, selected_features

if __name__ == "__main__":
    prepare_xgb_features()
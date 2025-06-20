import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def rf_feature_importance(X, y, top_n=15):
    """使用随机森林获取特征重要性"""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(top_n).index.tolist()

def create_rf_features(X):
    """创建随机森林特有的特征"""
    X_new = X.copy()
    # 添加特征组合
    if 'total_debt' in X.columns and 'income' in X.columns:
        X_new['debt_to_income'] = X['total_debt'] / (X['income'] + 1e-6)
    if 'loan_amnt' in X.columns and 'term' in X.columns:
        X_new['monthly_payment'] = X['loan_amnt'] / X['term']
    return X_new

def prepare_rf_features(file_path='process_data.csv'):
    """随机森林特征工程流程"""
    # 加载数据
    data = pd.read_csv(file_path)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    # 特征工程
    print("\n=== 随机森林特征工程 ===")
    # 1. 获取重要特征
    selected_features = rf_feature_importance(X, y)
    print(f"随机森林选择的重要特征 ({len(selected_features)}个):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 2. 创建特有特征
    X = create_rf_features(X)
    
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
    prepare_rf_features()
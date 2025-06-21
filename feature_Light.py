import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

def lightgbm_feature_importance(X, y, top_n=15):
    """Get feature importance using LightGBM"""
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(top_n).index.tolist()

def create_lightgbm_features(X):
    """Create LightGBM specific features"""
    X_new = X.copy()
    # Add categorical feature markers
    if 'grade' in X.columns:
        X_new['grade'] = X['grade'].astype('category')
    if 'sub_grade' in X.columns:
        X_new['sub_grade'] = X['sub_grade'].astype('category')
    return X_new

def prepare_lightgbm_features(file_path='process_data.csv'):
    """LightGBM feature engineering pipeline"""
    # Load data
    data = pd.read_csv(file_path)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    # Feature engineering
    print("\n=== LightGBM Feature Engineering ===")
    # 1. Get important features
    selected_features = lightgbm_feature_importance(X, y)
    print(f"LightGBM selected important features ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 2. Create specific features
    X = create_lightgbm_features(X)
    
    # 3. Apply feature selection
    X = X[selected_features]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nFinal feature dimension: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler, selected_features

if __name__ == "__main__":
    prepare_lightgbm_features()
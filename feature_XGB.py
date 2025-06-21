import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

def xgb_feature_importance(X, y, top_n=15):
    """Get feature importance using XGBoost"""
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(top_n).index.tolist()

def create_xgb_features(X):
    """Create XGBoost specific features"""
    X_new = X.copy()
    # Add debt-to-income ratio
    if 'total_debt' in X.columns and 'income' in X.columns:
        X_new['debt_to_income'] = X['total_debt'] / (X['income'] + 1e-6)
    # Add payment-to-income ratio
    if 'loan_amnt' in X.columns and 'income' in X.columns:
        X_new['payment_to_income'] = X['loan_amnt'] / (X['income'] + 1e-6)
    return X_new


# ============================================================================================================================


def prepare_xgb_features(file_path='process_data.csv'):
    """XGBoost feature engineering pipeline"""
    # Load data
    data = pd.read_csv(file_path)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    # Feature engineering
    print("\n=== XGBoost Feature Engineering ===")
    # 1. Get important features
    selected_features = xgb_feature_importance(X, y)
    print(f"XGBoost selected important features ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 2. Create specific features
    X = create_xgb_features(X)
    
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
    prepare_xgb_features()
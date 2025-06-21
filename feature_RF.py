import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def rf_feature_importance(X, y, top_n=15):
    """Get feature importance using Random Forest"""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(top_n).index.tolist()

def create_rf_features(X):
    """Create Random Forest specific features"""
    X_new = X.copy()
    # Add feature combinations
    if 'total_debt' in X.columns and 'income' in X.columns:
        X_new['debt_to_income'] = X['total_debt'] / (X['income'] + 1e-6)
    if 'loan_amnt' in X.columns and 'term' in X.columns:
        X_new['monthly_payment'] = X['loan_amnt'] / X['term']
    return X_new

def prepare_rf_features(file_path='process_data.csv'):
    """Random Forest feature engineering pipeline"""
    # Load data
    data = pd.read_csv(file_path)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    # Feature engineering
    print("\n=== Random Forest Feature Engineering ===")
    # 1. Get important features
    selected_features = rf_feature_importance(X, y)
    print(f"Random Forest selected important features ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 2. Create specific features
    X = create_rf_features(X)
    
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
    prepare_rf_features()
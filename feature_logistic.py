import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path='process_data.csv'):
    """Load and preprocess data"""
    data = pd.read_csv(file_path)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    return X, y


# ==================================================================================================

def feature_selection(X, y, k=10):
    """Feature selection"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

def create_polynomial_features(X, degree=2):
    """Create polynomial features"""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

# def create_interaction_features(X, selected_features, top_n=3):
#     """Create interaction features"""
#     X_interact = X.copy()
#     top_features = selected_features[:top_n]
#     for i in range(len(top_features)):
#         for j in range(i+1, len(top_features)):
#             col_name = f"{top_features[i]}_x_{top_features[j]}"
#             X_interact[col_name] = X[top_features[i]] * X[top_features[j]]
#     return X_interact

def prepare_features(X, y, k=10, degree=2, top_n=3):
    """Complete feature engineering pipeline"""
    # Feature selection
    X_selected, selected_features = feature_selection(X, y, k=k)
    
    # Output feature selection results
    print("\n=== Feature Engineering Details ===")
    print(f"Original feature count: {X.shape[1]}")
    print(f"Selected important features ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # Polynomial features
    X_poly = create_polynomial_features(X[selected_features], degree=degree)
    print(f"\nGenerated polynomial features: {X_poly.shape[1]-len(selected_features)}")
    
    # Combine features
    X_engineered = np.hstack([X_selected, X_poly])
    print(f"Total final features: {X_engineered.shape[1]}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n=== Feature Engineering Completed ===")
    print(f"Final feature dimension: {X_engineered.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler


# if __name__ == "__main__":
#     print("=== Feature Engineering Standalone Mode ===")
#     X, y = load_and_preprocess_data()
#     prepare_features(X, y)
#     print("\nNote: This is the standalone mode for feature engineering, only showing feature information.")
#     print("To train the model, please run logistic_regression_model.py")
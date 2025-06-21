import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

class FeaturePipeline:
    def __init__(self, model_type='random_forest'):
        """
        Initialize feature engineering pipeline
        :param model_type: Model type, determines which feature selection method to use
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.selector = None
        
        # Initialize feature selector based on model type
        if model_type == 'random_forest':
            self.base_model = RandomForestClassifier()
        elif model_type == 'xgboost':
            self.base_model = XGBClassifier()
        elif model_type == 'lightgbm':
            self.base_model = LGBMClassifier()
        elif model_type == 'logistic':
            self.base_model = LogisticRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit_transform(self, X, y):
        """
        Fit and transform features
        :param X: Feature data
        :param y: Label data
        :return: Transformed feature data
        """
        # Standardization (required for logistic regression, optional for other models)
        if self.model_type == 'logistic':
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
            
        # Feature selection
        self.selector = SelectFromModel(self.base_model, threshold='median')
        self.selector.fit(X_scaled, y)
        X_selected = self.selector.transform(X_scaled)
        
        return X_selected

    def transform(self, X):
        """
        Apply fitted transformations
        :param X: Feature data
        :return: Transformed feature data
        """
        if self.selector is None:
            raise RuntimeError("Must call fit_transform method first")
            
        # Standardization
        if self.model_type == 'logistic':
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
            
        # Feature selection
        X_selected = self.selector.transform(X_scaled)
        
        return X_selected
